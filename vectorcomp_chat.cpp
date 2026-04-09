// Vectorcomp V7 — Interactive Chat with Live KV Compression
// Links against llama.cpp to run a real model with Vectorcomp compression
// Shows real-time compression stats during chat

#include "vectorcomp_v7.hpp"
#include "llama.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <thread>

static float compute_cosine_similarity(const float* a, const float* b, int dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-9f || nb < 1e-9f) return 0.0f;
    return dot / std::sqrt(na * nb);
}

// Rotate K vector by RoPE - inverse of derotate_k
// Applies: [x0, x1] -> [x0*cos(t) - x1*sin(t), x0*sin(t) + x1*cos(t)]
static void rotate_k(const float* k_content, int position, float rope_freq_base, float* k_rotated, int head_dim) {
    for (int i = 0; i < head_dim / 2; ++i) {
        float theta = static_cast<float>(position) / std::pow(rope_freq_base, static_cast<float>(2 * i) / head_dim);
        float c = std::cos(theta);
        float s = std::sin(theta);
        float x0 = k_content[2 * i];
        float x1 = k_content[2 * i + 1];
        k_rotated[2 * i]     = x0 * c - x1 * s;
        k_rotated[2 * i + 1] = x0 * s + x1 * c;
    }
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f32;
    if (exp == 0) f32 = (sign << 31) | (mant << 13);
    else if (exp == 0x1f) f32 = (sign << 31) | (0x7f800000) | (mant << 13);
    else f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &f32, sizeof(float));
    return result;
}

struct CompressionStats {
    int total_tokens = 0;
    int total_ltm = 0;
    int total_stm = 0;
    int total_evicted = 0;
    float avg_k_sim = 1.0f;
    float avg_v_sim = 1.0f;
    size_t raw_kv_bytes = 0;
    size_t compressed_bytes = 0;
    double encode_ms = 0;
    double decode_ms = 0;
    // Similarity histogram
    int sim_hist_99_100 = 0;
    int sim_hist_95_99 = 0;
    int sim_hist_90_95 = 0;
    int sim_hist_85_90 = 0;
    int sim_hist_00_85 = 0;
    // Position validation
    int pos_min = 0;
    int pos_max = 0;
};

// Parse serialized KV cache and compress through Vectorcomp
CompressionStats compress_kv_cache(
    llama_context* ctx,
    const llama_model* model,
    std::vector<std::vector<std::unique_ptr<KVVectorcompV7>>>& vcomp_k,
    std::vector<std::vector<std::unique_ptr<KVVectorcompV7>>>& vcomp_v,
    std::vector<std::vector<uint32_t>>& k_ids,
    std::vector<std::vector<uint32_t>>& v_ids,
    std::vector<std::vector<int>>& pos_ids,
    float rope_freq_base,
    int ltm_slots,
    int stm_size) {

    CompressionStats stats;

    // Get model config
    int n_layer = llama_model_n_layer(model);
    int n_head_kv = llama_model_n_head_kv(model);
    int head_dim = llama_model_n_embd(model) / llama_model_n_head(model);
    int n_embd_k_gqa = n_head_kv * head_dim;
    int n_embd_v_gqa = n_head_kv * head_dim;

    // Extract KV cache via state API
    size_t state_size = llama_state_seq_get_size(ctx, 0);
    if (state_size == 0) {
        return stats;
    }

    std::vector<uint8_t> state_data(state_size);
    
    size_t copied = llama_state_seq_get_data(ctx, state_data.data(), state_size, 0);
    if (copied == 0) {
        return stats;
    }

    // Parse serialized format
    std::cout << "[compress] Parsing state..." << std::endl;
    const uint8_t* ptr = state_data.data();
    const uint8_t* end = ptr + copied;

    uint32_t n_stream = 0;
    if (ptr + 4 > end) {
        std::cout << "[compress] Not enough data for n_stream" << std::endl;
        return stats;
    }
    std::memcpy(&n_stream, ptr, 4); ptr += 4;
    std::cout << "[compress] n_stream=" << n_stream << std::endl;

    int total_cells = 0;
    int n_layers_with_kv = 0;

    for (uint32_t s = 0; s < n_stream && ptr < end; ++s) {
        std::cout << "[compress] Processing stream " << s << std::endl;
        uint32_t cell_count = 0;
        if (ptr + 4 > end) break;
        std::memcpy(&cell_count, ptr, 4); ptr += 4;
        std::cout << "[compress] cell_count=" << cell_count << std::endl;
        if (cell_count == 0) continue;

        total_cells = cell_count;

        // Skip metadata
        for (uint32_t c = 0; c < cell_count && ptr + 8 <= end; ++c) {
            int32_t pos = 0;
            uint32_t seq_count = 0;
            std::memcpy(&pos, ptr, 4); ptr += 4;
            std::memcpy(&seq_count, ptr, 4); ptr += 4;
            ptr += seq_count * 4;
        }

        std::cout << "[compress] After metadata, ptr offset=" << (ptr - state_data.data()) << std::endl;

        // Data header
        if (ptr + 8 > end) {
            std::cout << "[compress] Not enough data for header" << std::endl;
            break;
        }
        uint32_t v_trans = 0, n_layer_data = 0;
        std::memcpy(&v_trans, ptr, 4); ptr += 4;
        std::memcpy(&n_layer_data, ptr, 4); ptr += 4;
        std::cout << "[compress] v_trans=" << v_trans << ", n_layer_data=" << n_layer_data << std::endl;

        int n_layers_proc = std::min((int)n_layer_data, n_layer);
        std::cout << "[compress] n_layers_proc=" << n_layers_proc << std::endl;

        // Initialize Vectorcomp if needed
        std::cout << "[compress] Initializing vectorcomp..." << std::endl;
        if (vcomp_k.empty()) {
            vcomp_k.resize(n_layers_proc);
            vcomp_v.resize(n_layers_proc);
            k_ids.resize(n_layers_proc);
            v_ids.resize(n_layers_proc);
            pos_ids.resize(n_layers_proc);
            for (int l = 0; l < n_layers_proc; ++l) {
                k_ids[l].resize(cell_count * n_head_kv);
                v_ids[l].resize(cell_count * n_head_kv);
                pos_ids[l].resize(cell_count * n_head_kv);
                for (int h = 0; h < n_head_kv; ++h) {
                    vcomp_k[l].push_back(std::make_unique<KVVectorcompV7>(head_dim, ltm_slots, stm_size,
                                            rope_freq_base, 0.98f, 0.92f, 0.85f, 0.1f));
                    vcomp_v[l].push_back(std::make_unique<KVVectorcompV7>(head_dim, ltm_slots, stm_size,
                                            0.98f, 0.92f, 0.85f, 0.1f));
                }
            }
        }
        std::cout << "[compress] Vectorcomp init done" << std::endl;

        // Read K data - exact demo pattern: all K first, then all V
        std::cout << "[compress] Reading K data (demo pattern)..." << std::endl;
        
        std::vector<std::vector<float>> k_data(n_layers_proc);
        for (int l = 0; l < n_layers_proc && ptr + 12 <= end; ++l) {
            std::cout << "[compress] Reading K layer " << l << ", ptr offset=" << (ptr - state_data.data()) << std::endl;
            int32_t k_type = 0;
            uint64_t k_size_row = 0;
            std::memcpy(&k_type, ptr, 4); ptr += 4;
            std::memcpy(&k_size_row, ptr, 8); ptr += 8;
            std::cout << "[compress]   K type=" << k_type << ", size_row=" << k_size_row << std::endl;
            
            size_t k_data_size = (size_t)cell_count * k_size_row;
            if (ptr + k_data_size > end) {
                std::cout << "[compress]   K data not enough, breaking" << std::endl;
                break;
            }
            
            k_data[l].resize((size_t)cell_count * n_embd_k_gqa);
            if (k_type == 1) {
                const uint16_t* k_f16 = (const uint16_t*)ptr;
                for (size_t i = 0; i < k_data[l].size(); ++i)
                    k_data[l][i] = f16_to_f32(k_f16[i]);
            } else {
                std::memcpy(k_data[l].data(), ptr, k_data_size);
            }
            ptr += k_data_size;
            std::cout << "[compress]   K layer " << l << " done, size=" << k_data[l].size() << std::endl;
        }
        std::cout << "[compress] K data read done, k_data[0].size()=" << k_data[0].size() << std::endl;
        
        // Check if K data was read successfully
        if (k_data[0].empty()) {
            std::cerr << "[compress] ERROR: K data is empty, skipping compression" << std::endl;
            stats.total_tokens = 0;
            return stats;
        }
        
        // Read V data - handle both transposed and non-transposed formats
        std::cout << "[compress] Reading V data (v_trans=" << v_trans << ")..." << std::endl;
        std::vector<std::vector<float>> v_data(n_layers_proc);
        for (int l = 0; l < n_layers_proc && ptr + 12 <= end; ++l) {
            int32_t v_type = 0;
            std::memcpy(&v_type, ptr, 4); ptr += 4;
            
            if (v_trans == 0) {
                // Non-transposed: same format as K
                uint64_t v_size_row = 0;
                std::memcpy(&v_size_row, ptr, 8); ptr += 8;
                std::cout << "[compress]   V layer " << l << " type=" << v_type << ", size_row=" << v_size_row << std::endl;
                
                size_t v_data_size = (size_t)cell_count * v_size_row;
                if (ptr + v_data_size > end) {
                    std::cout << "[compress]   V data not enough, breaking" << std::endl;
                    break;
                }
                
                v_data[l].resize((size_t)cell_count * n_embd_v_gqa);
                if (v_type == 1) {
                    const uint16_t* v_f16 = (const uint16_t*)ptr;
                    for (size_t i = 0; i < v_data[l].size(); ++i)
                        v_data[l][i] = f16_to_f32(v_f16[i]);
                } else {
                    std::memcpy(v_data[l].data(), ptr, v_data_size);
                }
                ptr += v_data_size;
            } else {
                // Transposed format: v_type, v_size_el, n_embd_v_gqa, then data
                uint32_t v_size_el = 0, v_embd = 0;
                std::memcpy(&v_size_el, ptr, 4); ptr += 4;
                std::memcpy(&v_embd, ptr, 4); ptr += 4;
                std::cout << "[compress]   V layer " << l << " type=" << v_type << ", el_size=" << v_size_el << ", embd=" << v_embd << std::endl;
                
                // For transposed: read n_embd_v_gqa elements per cell (not v_size_row)
                v_data[l].resize((size_t)cell_count * n_embd_v_gqa);
                
                if (v_type == 1) {  // f16
                    for (uint32_t c = 0; c < cell_count; ++c) {
                        for (size_t i = 0; i < (size_t)n_embd_v_gqa; ++i) {
                            const uint16_t* v_f16 = (const uint16_t*)ptr;
                            v_data[l][c * n_embd_v_gqa + i] = f16_to_f32(v_f16[i]);
                        }
                        ptr += n_embd_v_gqa * 2;  // f16 = 2 bytes
                    }
                } else {  // f32
                    for (uint32_t c = 0; c < cell_count; ++c) {
                        std::memcpy(v_data[l].data() + (size_t)c * n_embd_v_gqa, ptr, (size_t)n_embd_v_gqa * sizeof(float));
                        ptr += n_embd_v_gqa * sizeof(float);
                    }
                }
            }
        }
        
        // Check if V data was read successfully - if not, skip compression
        if (v_data[0].empty()) {
            std::cerr << "[compress] ERROR: V data is empty, cannot compress (state format issue)" << std::endl;
            stats.total_tokens = 0;
            return stats;
        }
        
        n_layers_with_kv = n_layers_proc;

        // Compress through Vectorcomp
        auto t0 = std::chrono::high_resolution_clock::now();

        int total_ltm = 0, total_stm = 0, total_evicted = 0;
        float total_k_sim = 0, total_v_sim = 0;
        int checks = 0;
        int pos_min = 0, pos_max = 0;

        for (int l = 0; l < n_layers_proc; ++l) {
            for (int h = 0; h < n_head_kv; ++h) {
                for (uint32_t c = 0; c < cell_count; ++c) {
                    int token_idx = c * n_head_kv + h;
                    int k_offset = c * n_embd_k_gqa + h * head_dim;
                    int v_offset = c * n_embd_v_gqa + h * head_dim;
                    int position = static_cast<int>(c);

                    try {
                    k_ids[l][token_idx] = vcomp_k[l][h]->encode_shim_rope(
                        &k_data[l][(size_t)k_offset], &v_data[l][(size_t)v_offset], position);
                    pos_ids[l][token_idx] = position;
                    v_ids[l][token_idx] = vcomp_v[l][h]->encode_shim(
                        &v_data[l][(size_t)v_offset], &v_data[l][(size_t)v_offset]);
                    } catch (const std::exception& e) {
                        std::cerr << "[compress] Exception in encode: " << e.what() << std::endl;
                        throw;
                    } catch (...) {
                        std::cerr << "[compress] Unknown exception in encode" << std::endl;
                        throw;
                    }
                }
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        stats.encode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Decode and verify
        auto t2 = std::chrono::high_resolution_clock::now();
        std::vector<float> dk(head_dim), dv(head_dim), k_rotated(head_dim);
        for (int l = 0; l < n_layers_proc; ++l) {
            for (int h = 0; h < n_head_kv; ++h) {
                for (uint32_t c = 0; c < cell_count; ++c) {
                    int token_idx = c * n_head_kv + h;
                    int k_offset = c * n_embd_k_gqa + h * head_dim;
                    int v_offset = c * n_embd_v_gqa + h * head_dim;
                    int position = pos_ids[l][token_idx];

                    try {
                        vcomp_k[l][h]->decode_shim(&k_ids[l][token_idx], 1, dk.data(), dv.data());
                        // Re-rotate K for verification (content -> rotated)
                        rotate_k(dk.data(), position, rope_freq_base, k_rotated.data(), head_dim);
                        float k_sim = compute_cosine_similarity(
                            &k_data[l][(size_t)k_offset], k_rotated.data(), head_dim);
                        total_k_sim += k_sim;
                        checks++;
                        
                        // Histogram buckets
                        if (k_sim >= 0.99f) stats.sim_hist_99_100++;
                        else if (k_sim >= 0.95f) stats.sim_hist_95_99++;
                        else if (k_sim >= 0.90f) stats.sim_hist_90_95++;
                        else if (k_sim >= 0.85f) stats.sim_hist_85_90++;
                        else stats.sim_hist_00_85++;
                        
                        // Position validation
                        if (l == 0 && h == 0) {
                            if (position < stats.pos_min) stats.pos_min = position;
                            if (position > stats.pos_max) stats.pos_max = position;
                        }
                    } catch (...) { total_evicted++; }

                    try {
                        vcomp_v[l][h]->decode_shim(&v_ids[l][token_idx], 1, dv.data(), dv.data());
                        float v_sim = compute_cosine_similarity(
                            &v_data[l][(size_t)v_offset], dv.data(), head_dim);
                        total_v_sim += v_sim;
                    } catch (...) {}
                }
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        stats.decode_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        // Count LTM/STM
        for (int l = 0; l < n_layers_proc; ++l) {
            for (uint32_t id : k_ids[l]) {
                if ((id >> 31) & 1) total_stm++;
                else total_ltm++;
            }
        }

        stats.total_tokens = cell_count;
        stats.total_ltm = total_ltm;
        stats.total_stm = total_stm;
        stats.total_evicted = total_evicted;
        stats.avg_k_sim = checks > 0 ? total_k_sim / checks : 0;
        stats.avg_v_sim = checks > 0 ? total_v_sim / checks : 0;
        stats.raw_kv_bytes = (size_t)cell_count * n_layers_proc * (n_embd_k_gqa + n_embd_v_gqa) * 2;
        stats.compressed_bytes = (size_t)cell_count * n_layers_proc * n_head_kv * 2 * 4;
        stats.pos_min = pos_min;
        stats.pos_max = pos_max;
    }

    return stats;
}

void print_stats(const CompressionStats& stats, int turn) {
    std::cout << "\n  ┌─ Vectorcomp V7 Stats (Turn " << turn << ") ─────────────────────────┐" << std::endl;
    std::cout << "  │ Tokens: " << std::setw(4) << stats.total_tokens;
    std::cout << " │ LTM: " << std::setw(5) << stats.total_ltm;
    std::cout << " │ STM: " << std::setw(5) << stats.total_stm;
    std::cout << " │ Evicted: " << std::setw(3) << stats.total_evicted << " │" << std::endl;

    float k_sim_pct = stats.avg_k_sim * 100;
    float v_sim_pct = stats.avg_v_sim * 100;
    float compression = stats.compressed_bytes > 0 ? (float)stats.raw_kv_bytes / stats.compressed_bytes : 0;

    std::cout << "  │ K sim: " << std::fixed << std::setprecision(1) << k_sim_pct << "%";
    std::cout << " │ V sim: " << std::fixed << std::setprecision(1) << v_sim_pct << "%";
    std::cout << " │ Compression: " << std::fixed << std::setprecision(1) << compression << "x" << "    │" << std::endl;

    std::cout << "  │ Raw KV: " << std::fixed << std::setprecision(1) << (stats.raw_kv_bytes / 1024.0) << " KB";
    std::cout << " │ Compressed: " << std::fixed << std::setprecision(1) << (stats.compressed_bytes / 1024.0) << " KB";
    std::cout << " │ Encode: " << std::fixed << std::setprecision(1) << stats.encode_ms << "ms │" << std::endl;
    
    // Similarity histogram
    int total_hist = stats.sim_hist_99_100 + stats.sim_hist_95_99 + stats.sim_hist_90_95 + stats.sim_hist_85_90 + stats.sim_hist_00_85;
    if (total_hist > 0) {
        std::cout << "  │ Similarity Histogram (K):                                           │" << std::endl;
        std::cout << "  │ [0.99-1.00]: " << std::setw(3) << (stats.sim_hist_99_100 * 100 / total_hist) << "%";
        std::cout << " | [0.95-0.99]: " << std::setw(3) << (stats.sim_hist_95_99 * 100 / total_hist) << "%";
        std::cout << " | [0.90-0.95]: " << std::setw(3) << (stats.sim_hist_90_95 * 100 / total_hist) << "%";
        std::cout << " | [0.85-0.90]: " << std::setw(3) << (stats.sim_hist_85_90 * 100 / total_hist) << "%";
        std::cout << " | [<0.85]: " << std::setw(3) << (stats.sim_hist_00_85 * 100 / total_hist) << "%  │" << std::endl;
    }
    
    // Position validation
    std::cout << "  │ Position range: [min=" << stats.pos_min << ", max=" << stats.pos_max << "]";
    if (stats.pos_max - stats.pos_min > 0) std::cout << " (span=" << (stats.pos_max - stats.pos_min) << ")";
    std::cout << "                                      │" << std::endl;
    
    std::cout << "  └──────────────────────────────────────────────────────────┘" << std::endl;
}

int main(int argc, char** argv) {
    std::string model_path;
    int ltm_slots = 256;
    int stm_size = 256;
    float rope_freq_base = 130000.0f;

    bool vectorcomp_enabled = false;
    bool use_raw_prompt = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--ltm" && i + 1 < argc) ltm_slots = std::stoi(argv[++i]);
        else if (arg == "--stm" && i + 1 < argc) stm_size = std::stoi(argv[++i]);
        else if (arg == "--rope-freq" && i + 1 < argc) rope_freq_base = std::stof(argv[++i]);
        else if (arg == "--vectorcomp" || arg == "-vc") vectorcomp_enabled = true;
        else if (arg == "--no-vectorcomp") vectorcomp_enabled = false;
        else if (arg == "--raw-prompt") use_raw_prompt = true;
    }

    if (model_path.empty()) {
        std::cerr << "Usage: vectorcomp-chat -m <model.gguf> [--ltm N] [--stm N] [--rope-freq F] [--vectorcomp] [--raw-prompt]" << std::endl;
        return 1;
    }

    std::cout << "============================================================" << std::endl;
    std::cout << "  Vectorcomp V7 — Interactive Chat with Live Compression" << std::endl;
    std::cout << "============================================================" << std::endl;

    // Load model
    std::cout << "\nLoading model: " << model_path << std::endl;
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;  // Offload to GPU if available
    llama_model* model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    int n_ctx = 4096;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = 4096;
    cparams.n_ubatch = 4096;
    cparams.n_threads = 2;
    cparams.n_threads_batch = 2;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        llama_free_model(model);
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_layer = llama_model_n_layer(model);
    int n_head_kv = llama_model_n_head_kv(model);
    int head_dim = llama_model_n_embd(model) / llama_model_n_head(model);

    std::cout << "\nModel: " << llama_model_desc(model, nullptr, 0) << std::endl;
    std::cout << "  Layers: " << n_layer << ", KV heads: " << n_head_kv << ", head_dim: " << head_dim << std::endl;
    std::cout << "  Context: " << n_ctx << " tokens" << std::endl;
    // Get the chat template from the model
    const char * tmpl = llama_model_chat_template(model, nullptr);
    std::string template_name = "default";
    if (tmpl) {
        std::string t(tmpl);
        // Try to extract a short name from the template
        if (t.find("qwen") != std::string::npos) template_name = "Qwen";
        else if (t.find("chatml") != std::string::npos) template_name = "ChatML";
        else if (t.find("llama3") != std::string::npos || t.find("llama-3") != std::string::npos) template_name = "Llama3";
        else if (t.find("mistral") != std::string::npos) template_name = "Mistral";
        else if (t.find("phi") != std::string::npos) template_name = "Phi";
        else if (t.find("{%") != std::string::npos) template_name = "Jinja";  // Raw Jinja template
        else if (t.length() < 50) template_name = t;  // Short name
    }
    
    std::cout << "  Chat template: " << template_name << std::endl;
    std::cout << "\nType your messages (type 'quit' to exit):\n" << std::endl;

    // Create sampler chain - simplified
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Vectorcomp instances (recreated each turn for simplicity)
    std::vector<std::vector<std::unique_ptr<KVVectorcompV7>>> vcomp_k, vcomp_v;
    std::vector<std::vector<uint32_t>> k_ids, v_ids;
    std::vector<std::vector<int>> pos_ids;

    int turn = 0;
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(8192);

    auto add_message = [&](const char* role, const std::string& content) {
        char* copy = strdup(content.c_str());
        messages.push_back({role, copy});
    };

    auto free_messages = [&]() {
        for (auto & msg : messages) {
            free((void*)msg.content);
        }
        messages.clear();
    };

    // Keep just the last few messages to prevent template bloat
    while (messages.size() > 6) {
        free((void*)messages[0].content);
        messages.erase(messages.begin());
    }

    std::string line;
    while (true) {
        std::cout << "\nYou: ";
        std::getline(std::cin, line);
        if (line == "quit" || line == "exit") break;
        if (line.empty()) continue;

        turn++;

        std::string prompt;
        
        if (use_raw_prompt) {
            // Use raw prompt without chat template
            prompt = line;
            
            // Tokenize the prompt
            std::vector<llama_token> prompt_tokens(4096);
            int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                           prompt_tokens.data(), (int)prompt_tokens.size(), false, false);
            if (n_tokens < 0) {
                std::cerr << "Failed to tokenize prompt" << std::endl;
                free_messages();
                continue;
            }
            prompt_tokens.resize(n_tokens);

            // Trim conversation if too long
            int max_tokens = n_ctx - 128;
            if ((int)prompt_tokens.size() > max_tokens) {
                prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.begin() + (int)prompt_tokens.size() - max_tokens);
            }

            // Run inference
            llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int)prompt_tokens.size());
            if (llama_decode(ctx, batch) != 0) {
                std::cerr << "Decode failed" << std::endl;
                free_messages();
                continue;
            }
        } else {
            // Add user message and format with chat template
            add_message("user", line);
            
            int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
            if (new_len > (int)formatted.size()) {
                formatted.resize(new_len + 1);
                new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
            }
            if (new_len < 0) {
                std::cerr << "Failed to apply chat template" << std::endl;
                free_messages();
                continue;
            }
            formatted[new_len] = '\0';

            // Get the full formatted prompt
            prompt = std::string(formatted.begin(), formatted.begin() + new_len);

            // Tokenize the prompt
            std::vector<llama_token> prompt_tokens(4096);
            int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                           prompt_tokens.data(), (int)prompt_tokens.size(), false, false);
            if (n_tokens < 0) {
                std::cerr << "Failed to tokenize prompt" << std::endl;
                free_messages();
                continue;
            }
            prompt_tokens.resize(n_tokens);

            // Trim conversation if too long - keep the LAST (n_ctx - 128) tokens
            int max_tokens = n_ctx - 128;
            if ((int)prompt_tokens.size() > max_tokens) {
                prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.begin() + (int)prompt_tokens.size() - max_tokens);
            }

            // Run inference
            llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int)prompt_tokens.size());
            if (llama_decode(ctx, batch) != 0) {
                std::cerr << "Decode failed" << std::endl;
                free_messages();
                continue;
            }
        }

        // Sample next token
        std::string response;
        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

        // Check for EOS using the model's actual EOS token
        while (true) {
            // Print token if not EOS
            if (llama_vocab_is_eog(vocab, new_token)) {
                break;
            }
            
            std::vector<char> buf(256);
            int n = llama_token_to_piece(vocab, new_token, buf.data(), (int)buf.size(), 0, true);
            if (n > 0) {
                std::cout << std::string(buf.data(), n) << std::flush;
                response.append(buf.data(), n);
            }
            
            // Check for early termination - don't continue if response is too long
            if (response.length() > 512) break;
            
            // Generate next token
            llama_batch batch2 = llama_batch_get_one(&new_token, 1);
            if (llama_decode(ctx, batch2) != 0) break;
            
            new_token = llama_sampler_sample(smpl, ctx, -1);
            
            // Also check for common EOS patterns in token string
            std::vector<char> next_buf(256);
            int next_n = llama_token_to_piece(vocab, new_token, next_buf.data(), (int)next_buf.size(), 0, true);
            if (next_n > 0) {
                std::string token_str(next_buf.data(), next_n);
                if (token_str == "<|im_end|>" || token_str == "<|endoftext|>") {
                    break;
                }
            }
        }
        std::cout << std::endl;

        // Add assistant response to messages (use raw response, not template-formatted)
        add_message("assistant", response);
        
        // Debug: show what the formatted prompt looks like now
        std::vector<char> debug_formatted(8192);
        int debug_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, debug_formatted.data(), debug_formatted.size());
        if (debug_len > 0 && debug_len < (int)debug_formatted.size()) {
            debug_formatted[debug_len] = '\0';
            std::cout << "\n[Debug] Formatted prompt:\n" << debug_formatted.data() << "\n---END---\n" << std::endl;
        }
        
        // Trim old messages to prevent template bloat
        while (messages.size() > 6) {
            free((void*)messages[0].content);
            messages.erase(messages.begin());
        }

        // Compress KV cache through Vectorcomp
        if (vectorcomp_enabled) {
            std::cout << "[Debug] vectorcomp_enabled=" << vectorcomp_enabled << ", vcomp_k.size()=" << vcomp_k.size() << std::endl;
            // Clear state to prevent cross-turn reuse issues that cause crashes on turn 2+
            vcomp_k.clear();
            vcomp_v.clear();
            k_ids.clear();
            v_ids.clear();
            pos_ids.clear();
            auto stats = compress_kv_cache(ctx, model, vcomp_k, vcomp_v, k_ids, v_ids, pos_ids,
                                            rope_freq_base, ltm_slots, stm_size);
            print_stats(stats, turn);
        } else {
            std::cout << "[Debug] Compression disabled (use --vectorcomp to enable)" << std::endl;
        }
    }

    // Free message contents
    for (auto & msg : messages) {
        free((void*)msg.content);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);
    std::cout << "\nGoodbye!" << std::endl;
    return 0;
}
