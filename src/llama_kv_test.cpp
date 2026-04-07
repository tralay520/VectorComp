#include "vectorcomp.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
#include <cstring>
#include <numeric>

// llama.cpp KV Cache Integration Test for Vectorcomp V6
// Simulates the exact memory layout and KV variation patterns of a real model
// No llama.cpp linking required — standalone test with realistic parameters

// Minimal ggml_type enum (subset we need)
enum ggml_type {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q4_0 = 2,
};

static size_t ggml_type_size(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_F16: return 2;
        case GGML_TYPE_Q8_0: return 1;
        case GGML_TYPE_Q4_0: return 1;
        default: return 4;
    }
}

// ============================================================
// Simulated llama.cpp model configuration (Qwen2.5 0.5B)
// ============================================================
struct SimulatedModelConfig {
    std::string name;
    int n_vocab;
    int n_embd;
    int n_layer;
    int n_head;
    int n_head_kv;
    int head_dim;
    int ctx_size;
    ggml_type type_k;
    ggml_type type_v;
};

// ============================================================
// llama.cpp KV Cache Simulator
// Replicates the exact memory layout: 3D tensor [n_embd_k_gqa, kv_size, n_stream]
// ============================================================
class LlamaKVCacheSim {
    int n_embd_k_gqa;  // n_head_kv * head_dim
    int n_embd_v_gqa;
    int kv_size;
    int n_layers;
    ggml_type type_k, type_v;

    // Raw KV buffers: one per layer, laid out exactly like llama.cpp
    // K: [n_embd_k_gqa, kv_size] — row-major, element size = type_size
    // V: [n_embd_v_gqa, kv_size] — same (non-transposed for simplicity)
    std::vector<std::vector<uint8_t>> k_layers, v_layers;

public:
    LlamaKVCacheSim(int n_layer, int n_head_kv, int head_dim,
                    int kv_size, ggml_type tk, ggml_type tv)
        : n_embd_k_gqa(n_head_kv * head_dim),
          n_embd_v_gqa(n_head_kv * head_dim),
          kv_size(kv_size), n_layers(n_layer), type_k(tk), type_v(tv) {

        size_t k_layer_size = static_cast<size_t>(n_embd_k_gqa) * kv_size * ggml_type_size(type_k);
        size_t v_layer_size = static_cast<size_t>(n_embd_v_gqa) * kv_size * ggml_type_size(type_v);

        k_layers.resize(n_layer, std::vector<uint8_t>(k_layer_size, 0));
        v_layers.resize(n_layer, std::vector<uint8_t>(v_layer_size, 0));
    }

    // Get raw pointer to K data for a specific layer and token position
    float* get_k_f32(int layer, int token_pos) {
        size_t offset = static_cast<size_t>(token_pos) * n_embd_k_gqa;
        // In real llama.cpp this would be quantized, but we simulate F32 for this test
        return reinterpret_cast<float*>(k_layers[layer].data()) + offset;
    }

    float* get_v_f32(int layer, int token_pos) {
        size_t offset = static_cast<size_t>(token_pos) * n_embd_v_gqa;
        return reinterpret_cast<float*>(v_layers[layer].data()) + offset;
    }

    // Total memory used (simulating F16 storage like llama.cpp default)
    size_t total_bytes() const {
        size_t k_layer_size = static_cast<size_t>(n_embd_k_gqa) * kv_size * ggml_type_size(type_k);
        size_t v_layer_size = static_cast<size_t>(n_embd_v_gqa) * kv_size * ggml_type_size(type_v);
        return static_cast<size_t>(n_layers) * (k_layer_size + v_layer_size);
    }

    int get_n_embd_k_gqa() const { return n_embd_k_gqa; }
    int get_n_embd_v_gqa() const { return n_embd_v_gqa; }
    int get_kv_size() const { return kv_size; }
    int get_n_layers() const { return n_layers; }
};

// ============================================================
// Realistic KV Vector Generator
// Simulates what a real transformer produces:
// - Position-dependent K vectors (RoPE-like variation)
// - Token-embedding-dependent V vectors
// - Layer-dependent variation (deeper layers = more abstract)
// ============================================================
class RealisticKVGenerator {
    int head_dim;
    int n_embd_k_gqa;
    int n_embd_v_gqa;
    int n_layers;
    std::mt19937 gen;

    // Pre-generated base vectors per token per layer (flat arrays)
    std::vector<float> base_k_per_token_per_layer;
    std::vector<float> base_v_per_token_per_layer;

    // RoPE-like rotation tables
    std::vector<std::vector<float>> rope_cos, rope_sin;

public:
    RealisticKVGenerator(int head_dim, int n_head_kv, int n_layers,
                         int vocab_size, uint32_t seed = 12345)
        : head_dim(head_dim),
          n_embd_k_gqa(n_head_kv * head_dim),
          n_embd_v_gqa(n_head_kv * head_dim),
          n_layers(n_layers), gen(seed) {

        std::normal_distribution<float> ndist(0.0f, 1.0f / std::sqrt(static_cast<float>(head_dim)));

        // Generate base K/V per token per layer
        int total_vectors = vocab_size * n_layers;
        base_k_per_token_per_layer.resize(static_cast<size_t>(total_vectors) * n_embd_k_gqa);
        base_v_per_token_per_layer.resize(static_cast<size_t>(total_vectors) * n_embd_v_gqa);

        for (int v = 0; v < vocab_size; ++v) {
            for (int l = 0; l < n_layers; ++l) {
                // Deeper layers have more variation
                float layer_scale = 1.0f + 0.5f * static_cast<float>(l) / n_layers;

                for (int d = 0; d < n_embd_k_gqa; ++d) {
                    size_t idx = static_cast<size_t>(v * n_layers + l) * n_embd_k_gqa + d;
                    base_k_per_token_per_layer[idx] = ndist(gen) * layer_scale;
                }
                for (int d = 0; d < n_embd_v_gqa; ++d) {
                    size_t idx = static_cast<size_t>(v * n_layers + l) * n_embd_v_gqa + d;
                    base_v_per_token_per_layer[idx] = ndist(gen) * layer_scale;
                }
            }
        }

        // Pre-compute RoPE-like rotation tables for positions 0..2047
        int max_pos = 2048;
        rope_cos.resize(max_pos);
        rope_sin.resize(max_pos);
        float base = 10000.0f;
        for (int pos = 0; pos < max_pos; ++pos) {
            rope_cos[pos].resize(n_embd_k_gqa);
            rope_sin[pos].resize(n_embd_k_gqa);
            for (int d = 0; d < n_embd_k_gqa; ++d) {
                int dim_pair = d / 2;
                float freq = 1.0f / std::pow(base, static_cast<float>(dim_pair * 2) / n_embd_k_gqa);
                float theta = static_cast<float>(pos) * freq;
                rope_cos[pos][d] = std::cos(theta);
                rope_sin[pos][d] = std::sin(theta);
            }
        }
    }

    // Generate K vector for a specific token, position, and layer
    // Applies RoPE-like rotation + small noise (like real model inference)
    void generate_k(int token_id, int position, int layer, float* out) const {
        const float* base = &base_k_per_token_per_layer[
            static_cast<size_t>(token_id * n_layers + layer) * n_embd_k_gqa];

        // Apply RoPE-like rotation
        for (int d = 0; d < n_embd_k_gqa; d += 2) {
            float x0 = base[d], x1 = base[d + 1];
            float c = rope_cos.at(position)[d];
            float s = rope_sin.at(position)[d];
            out[d]     = x0 * c - x1 * s;
            out[d + 1] = x0 * s + x1 * c;
        }

        // Add tiny inference noise (quantization, numerical precision)
        std::mt19937 noise_gen(static_cast<uint32_t>(token_id * 100000 + position * 1000 + layer * 100 + 7));
        std::normal_distribution<float> noise(0.0f, 0.001f);
        for (int d = 0; d < n_embd_k_gqa; ++d) {
            out[d] += noise(noise_gen);
        }
    }

    // Generate V vector (no RoPE, but position-independent + small noise)
    void generate_v(int token_id, int position, int layer, float* out) const {
        const float* base = &base_v_per_token_per_layer[
            static_cast<size_t>(token_id * n_layers + layer) * n_embd_v_gqa];

        std::memcpy(out, base, static_cast<size_t>(n_embd_v_gqa) * sizeof(float));

        // Small inference noise
        std::mt19937 noise_gen(static_cast<uint32_t>(token_id * 100000 + position * 1000 + layer * 100 + 13));
        std::normal_distribution<float> noise(0.0f, 0.002f);
        for (int d = 0; d < n_embd_v_gqa; ++d) {
            out[d] += noise(noise_gen);
        }
    }
};

// ============================================================
// Vectorcomp KV Cache Wrapper
// Wraps Vectorcomp to act as a drop-in replacement for llama.cpp's KV cache
// ============================================================
class VectorcompKVCache {
    int n_layers;
    int head_dim;
    int n_embd_k_gqa;
    int n_embd_v_gqa;
    std::vector<std::unique_ptr<KVVectorcompV6>> vcomp_per_layer;

public:
    VectorcompKVCache(int n_layer, int n_head_kv, int head_dim,
                      int ltm_slots, int stm_size)
        : n_layers(n_layer), head_dim(head_dim),
          n_embd_k_gqa(n_head_kv * head_dim),
          n_embd_v_gqa(n_head_kv * head_dim) {

        vcomp_per_layer.reserve(n_layer);
        for (int l = 0; l < n_layer; ++l) {
            vcomp_per_layer.push_back(std::make_unique<KVVectorcompV6>(head_dim * n_head_kv, ltm_slots, stm_size));
        }
    }

    // Encode a single token's K/V for a specific layer
    uint32_t encode(int layer, const float* k, const float* v) {
        return vcomp_per_layer[layer]->encode_shim(k, v);
    }

    void decode(int layer, const uint32_t* ids, int seq_len, float* out_k, float* out_v) {
        vcomp_per_layer[layer]->decode_shim(ids, seq_len, out_k, out_v);
    }

    size_t get_active_cb_count(int layer) const {
        return vcomp_per_layer[layer]->get_active_cb_count();
    }

    size_t total_bytes() const {
        size_t per_layer = static_cast<size_t>(vcomp_per_layer[0]->get_active_cb_count()) *
                          n_embd_k_gqa * sizeof(float) * 2; // K+V codebook
        // Plus raw buffer (approximate)
        return per_layer * n_layers;
    }
};

// ============================================================
// Benchmark: Fill KV cache with realistic data, compress with Vectorcomp
// ============================================================
void run_llama_benchmark(const std::string& name,
                          const SimulatedModelConfig& config,
                          const std::vector<int>& token_sequence,
                          int ltm_slots, int stm_size,
                          int effective_vocab = 0) {
    std::cout << "\n=== " << name << " ===" << std::endl;
    std::cout << "  Model: " << config.name << std::endl;
    std::cout << "  Layers: " << config.n_layer
              << ", n_embd_k_gqa: " << config.n_head_kv * config.head_dim
              << ", head_dim: " << config.head_dim
              << ", n_head_kv: " << config.n_head_kv << std::endl;
    std::cout << "  Sequence length: " << token_sequence.size() << " tokens" << std::endl;
    std::cout << "  Vectorcomp config: LTM=" << ltm_slots << " STM=" << stm_size << std::endl;

    int n_embd_k_gqa = config.n_head_kv * config.head_dim;
    int n_embd_v_gqa = config.n_head_kv * config.head_dim;
    int seq_len = static_cast<int>(token_sequence.size());
    int n_layers = config.n_layer;

    // Use effective_vocab (tokens we actually use) not full model vocab
    int vocab = effective_vocab > 0 ? effective_vocab : config.n_vocab;

    // Create realistic KV generator
    RealisticKVGenerator gen(config.head_dim, config.n_head_kv, n_layers,
                             vocab, 42);

    // Create simulated llama.cpp KV cache
    LlamaKVCacheSim llama_cache(n_layers, config.n_head_kv, config.head_dim,
                                config.ctx_size, GGML_TYPE_F16, GGML_TYPE_F16);

    // Create Vectorcomp KV cache
    VectorcompKVCache vc_cache(n_layers, config.n_head_kv, config.head_dim,
                               ltm_slots, stm_size);

    // Buffers for K/V
    std::vector<float> k_buf(n_embd_k_gqa);
    std::vector<float> v_buf(n_embd_v_gqa);

    // ---- Phase 1: Fill llama.cpp KV cache (baseline) ----
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int pos = 0; pos < seq_len; ++pos) {
        int token = token_sequence[pos];
        for (int l = 0; l < n_layers; ++l) {
            gen.generate_k(token, pos, l, llama_cache.get_k_f32(l, pos));
            gen.generate_v(token, pos, l, llama_cache.get_v_f32(l, pos));
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double llama_fill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- Phase 2: Compress through Vectorcomp ----
    std::vector<std::vector<uint32_t>> vc_ids(n_layers, std::vector<uint32_t>(seq_len));
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int pos = 0; pos < seq_len; ++pos) {
        int token = token_sequence[pos];
        for (int l = 0; l < n_layers; ++l) {
            gen.generate_k(token, pos, l, k_buf.data());
            gen.generate_v(token, pos, l, v_buf.data());
            vc_ids[l][pos] = vc_cache.encode(l, k_buf.data(), v_buf.data());
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double vc_encode_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // ---- Phase 3: Decode and verify quality ----
    std::vector<float> decoded_k(n_embd_k_gqa);
    std::vector<float> decoded_v(n_embd_v_gqa);

    float total_k_sim = 0.0f, total_v_sim = 0.0f;
    float min_k_sim = 1.0f, min_v_sim = 1.0f;
    int total_checks = 0;
    int evicted = 0;

    auto t4 = std::chrono::high_resolution_clock::now();
    for (int l = 0; l < n_layers; ++l) {
        for (int pos = 0; pos < seq_len; ++pos) {
            try {
                vc_cache.decode(l, &vc_ids[l][pos], 1, decoded_k.data(), decoded_v.data());

                // Generate original for comparison
                gen.generate_k(token_sequence[pos], pos, l, k_buf.data());
                gen.generate_v(token_sequence[pos], pos, l, v_buf.data());

                // Cosine similarity
                float k_dot = 0, k_na = 0, k_nb = 0;
                float v_dot = 0, v_na = 0, v_nb = 0;
                for (int d = 0; d < n_embd_k_gqa; ++d) {
                    k_dot += decoded_k[d] * k_buf[d];
                    k_na += decoded_k[d] * decoded_k[d];
                    k_nb += k_buf[d] * k_buf[d];
                    v_dot += decoded_v[d] * v_buf[d];
                    v_na += decoded_v[d] * decoded_v[d];
                    v_nb += v_buf[d] * v_buf[d];
                }
                float k_sim = (k_na < 1e-9f || k_nb < 1e-9f) ? 0.0f : k_dot / std::sqrt(k_na * k_nb);
                float v_sim = (v_na < 1e-9f || v_nb < 1e-9f) ? 0.0f : v_dot / std::sqrt(v_na * v_nb);

                total_k_sim += k_sim;
                total_v_sim += v_sim;
                if (k_sim < min_k_sim) min_k_sim = k_sim;
                if (v_sim < min_v_sim) min_v_sim = v_sim;
                total_checks++;
            } catch (const std::runtime_error&) {
                evicted++;
            }
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    double vc_decode_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    // ---- Stats ----
    int total_ltm = 0, total_stm = 0;
    for (int l = 0; l < n_layers; ++l) {
        for (uint32_t id : vc_ids[l]) {
            if ((id >> 31) & 1) total_stm++;
            else total_ltm++;
        }
    }
    int total_tokens = seq_len * n_layers;

    // Memory comparison
    size_t llama_bytes = llama_cache.total_bytes();
    // Vectorcomp: codebook + raw buffer per layer
    size_t vc_codebook = static_cast<size_t>(ltm_slots) * n_embd_k_gqa * sizeof(float) * 2; // K+V
    size_t vc_raw = static_cast<size_t>(stm_size) * n_embd_k_gqa * sizeof(float) * 2;
    size_t vc_meta = static_cast<size_t>(ltm_slots) * 16;
    size_t vc_per_layer = vc_codebook + vc_raw + vc_meta;
    size_t vc_total = vc_per_layer * n_layers;

    // Compressed IDs memory
    size_t ids_bytes = static_cast<size_t>(total_tokens) * sizeof(uint32_t);

    float avg_k_sim = total_checks > 0 ? total_k_sim / total_checks : 0.0f;
    float avg_v_sim = total_checks > 0 ? total_v_sim / total_checks : 0.0f;

    std::cout << "\n  Compression Stats:" << std::endl;
    std::cout << "    LTM hits: " << total_ltm << "/" << total_tokens << std::endl;
    std::cout << "    STM writes: " << total_stm << "/" << total_tokens << std::endl;
    std::cout << "    STM evictions: " << evicted << std::endl;

    std::cout << "\n  Memory Comparison:" << std::endl;
    std::cout << "    llama.cpp KV cache:  " << std::fixed << std::setprecision(1)
              << (llama_bytes / 1024.0) << " KB" << std::endl;
    std::cout << "    Vectorcomp overhead: " << std::fixed << std::setprecision(1)
              << (vc_total / 1024.0) << " KB" << std::endl;
    std::cout << "    Compressed IDs:      " << std::fixed << std::setprecision(1)
              << (ids_bytes / 1024.0) << " KB" << std::endl;
    std::cout << "    ID compression:      " << std::fixed << std::setprecision(1)
              << (static_cast<float>(llama_bytes) / ids_bytes) << "x" << std::endl;

    std::cout << "\n  Quality (decoded vs original):" << std::endl;
    std::cout << "    K similarity: avg=" << std::fixed << std::setprecision(4) << avg_k_sim
              << " min=" << min_k_sim << std::endl;
    std::cout << "    V similarity: avg=" << std::fixed << std::setprecision(4) << avg_v_sim
              << " min=" << min_v_sim << std::endl;

    std::cout << "\n  Timing:" << std::endl;
    std::cout << "    llama.cpp fill:  " << std::fixed << std::setprecision(2) << llama_fill_ms << " ms" << std::endl;
    std::cout << "    VC encode:       " << vc_encode_ms << " ms" << std::endl;
    std::cout << "    VC decode:       " << vc_decode_ms << " ms" << std::endl;
}

// ============================================================
// Main: Run benchmarks with realistic model configs
// ============================================================
int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "  Vectorcomp V6 — llama.cpp KV Cache Integration Test" << std::endl;
    std::cout << "============================================================" << std::endl;

    // Model configs (real small models that could run on modest hardware)
    // Note: effective_vocab is what the generator actually pre-computes
    SimulatedModelConfig qwen_05b = {
        "Qwen2.5 0.5B",
        151936, 896, 24, 14, 2, 64, 2048,
        GGML_TYPE_F16, GGML_TYPE_F16
    };

    SimulatedModelConfig llama_1b = {
        "Llama 3.2 1B",
        128256, 2048, 16, 32, 8, 64, 2048,
        GGML_TYPE_F16, GGML_TYPE_F16
    };

    // Generate token sequences with realistic patterns
    std::vector<int> short_repeated;
    std::vector<int> short_varied;
    std::vector<int> long_mixed;

    std::mt19937 gen(99999);
    std::uniform_int_distribution<int> dist_vocab(1, 50);

    // Short repeated: 30 tokens, only 10 unique
    for (int i = 0; i < 30; ++i) short_repeated.push_back(1 + (i % 10));

    // Short varied: 30 tokens, all different
    for (int i = 0; i < 30; ++i) short_varied.push_back(i + 1);

    // Long mixed: 100 tokens, ~20 unique with realistic repetition patterns
    for (int i = 0; i < 100; ++i) {
        if (i % 5 == 0) long_mixed.push_back(dist_vocab(gen));
        else long_mixed.push_back(long_mixed.back() > 0 ? long_mixed.back() - 1 : 1);
    }

    // Effective vocab = max token ID we actually use (not the full model vocab)
    int effective_vocab = 0;
    for (int t : short_repeated) effective_vocab = std::max(effective_vocab, t + 1);
    for (int t : short_varied) effective_vocab = std::max(effective_vocab, t + 1);
    for (int t : long_mixed) effective_vocab = std::max(effective_vocab, t + 1);

    // ---- Test 1: Small model, repeated tokens (best case) ----
    run_llama_benchmark("Test 1: Qwen2.5 0.5B — Repeated Tokens (Best Case)",
                        qwen_05b, short_repeated, 32, 64, effective_vocab);

    // ---- Test 2: Small model, varied tokens (worst case) ----
    run_llama_benchmark("Test 2: Qwen2.5 0.5B — Varied Tokens (Worst Case)",
                        qwen_05b, short_varied, 32, 64, effective_vocab);

    // ---- Test 3: Small model, long mixed sequence ----
    run_llama_benchmark("Test 3: Qwen2.5 0.5B — Long Mixed Sequence",
                        qwen_05b, long_mixed, 64, 128, effective_vocab);

    // ---- Test 4: Larger model (Llama 1B), short repeated ----
    run_llama_benchmark("Test 4: Llama 3.2 1B — Repeated Tokens",
                        llama_1b, short_repeated, 32, 64, effective_vocab);

    // ---- Test 5: Memory scaling analysis ----
    std::cout << "\n=== Test 5: Memory Scaling Analysis ===" << std::endl;
    std::cout << "  Comparing KV cache sizes across model scales:" << std::endl;
    std::cout << std::endl;
    std::cout << "  Model              | KV Cache (KB) | IDs (KB) | Compression" << std::endl;
    std::cout << "  -------------------|---------------|----------|------------" << std::endl;

    struct ScaleTest { std::string name; int n_layer; int n_head_kv; int head_dim; int ctx; };
    std::vector<ScaleTest> scales = {
        {"Tiny (125M)",     12, 2,  64, 512},
        {"Small (350M)",    24, 4,  64, 1024},
        {"Medium (1B)",     16, 8,  64, 2048},
        {"Large (3B)",      28, 8, 128, 4096},
    };

    for (const auto& s : scales) {
        int n_embd_k = s.n_head_kv * s.head_dim;
        size_t raw_kv = static_cast<size_t>(s.n_layer) * n_embd_k * s.ctx * 4; // F32 for comparison
        size_t ids = static_cast<size_t>(s.ctx) * s.n_layer * 4; // uint32_t per token per layer
        float ratio = static_cast<float>(raw_kv) / ids;

        std::cout << "  " << std::left << std::setw(19) << s.name
                  << " | " << std::right << std::setw(13) << std::fixed << std::setprecision(1)
                  << (raw_kv / 1024.0)
                  << " | " << std::setw(8) << std::fixed << std::setprecision(1)
                  << (ids / 1024.0)
                  << " | " << std::setw(10) << std::fixed << std::setprecision(1)
                  << ratio << "x" << std::endl;
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << "  llama.cpp KV Cache Integration Test Complete!" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
