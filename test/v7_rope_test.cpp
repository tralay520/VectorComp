// Vectorcomp V7 — RoPE-Aware Rehydration Test
// Tests the derotate_k mechanism on real model KV data from SmolLM2 1.7B
// This is a standalone test (no llama.cpp linking needed) that simulates
// the exact RoPE patterns a real model produces.

#include "../src/vectorcomp_v7.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>

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

// Apply RoPE rotation to a K vector (simulates what the model does)
static void apply_rope(const float* k_content, int position, float* k_rotated,
                        int head_dim, float rope_freq_base) {
    for (int i = 0; i < head_dim / 2; ++i) {
        float freq = 1.0f / std::pow(rope_freq_base, static_cast<float>(2 * i) / head_dim);
        float theta = static_cast<float>(position) * freq;
        float c = std::cos(theta);
        float s = std::sin(theta);
        float x0 = k_content[2 * i];
        float x1 = k_content[2 * i + 1];
        k_rotated[2 * i]     = x0 * c - x1 * s;
        k_rotated[2 * i + 1] = x0 * s + x1 * c;
    }
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "  Vectorcomp V7 — RoPE-Aware Rehydration Test" << std::endl;
    std::cout << "============================================================" << std::endl;

    // SmolLM2 1.7B configuration
    int head_dim = 64;
    int n_heads = 32;
    int n_layers = 24;
    int n_tokens = 256;
    float rope_freq_base = 130000.0f;
    int ltm_slots = 256;
    int stm_size = 256;

    std::cout << "\nConfig: head_dim=" << head_dim << ", n_heads=" << n_heads
              << ", n_layers=" << n_layers << ", tokens=" << n_tokens << std::endl;
    std::cout << "  RoPE freq_base=" << rope_freq_base << std::endl;
    std::cout << "  LTM=" << ltm_slots << ", STM=" << stm_size << std::endl;

    // Generate realistic content-only K vectors (same token repeated = same content)
    // This simulates: "the the the the..." where each "the" has the same semantic content
    // but different RoPE positions
    std::mt19937 gen(42);
    std::normal_distribution<float> ndist(0.0f, 1.0f / std::sqrt(static_cast<float>(head_dim)));

    // Create 10 unique "token concepts" — each with a fixed content vector
    int n_concepts = 10;
    std::vector<std::vector<float>> concept_k(n_concepts, std::vector<float>(head_dim));
    for (int c = 0; c < n_concepts; ++c) {
        for (int d = 0; d < head_dim; ++d) {
            concept_k[c][d] = ndist(gen);
        }
        // Normalize
        float norm = 0.0f;
        for (float f : concept_k[c]) norm += f * f;
        norm = std::sqrt(norm);
        if (norm > 1e-9f) for (float& f : concept_k[c]) f /= norm;
    }

    // Generate token sequence: repeat concepts across positions
    // Token i uses concept (i % n_concepts) at position i
    std::vector<int> token_concepts(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        token_concepts[i] = i % n_concepts;
    }

    // Generate RoPE-rotated K vectors (what the model actually produces)
    std::vector<std::vector<float>> k_rotated(n_tokens, std::vector<float>(head_dim));
    std::vector<std::vector<float>> v_data(n_tokens, std::vector<float>(head_dim));
    for (int i = 0; i < n_tokens; ++i) {
        int concept = token_concepts[i];
        apply_rope(concept_k[concept].data(), i, k_rotated[i].data(), head_dim, rope_freq_base);
        // V is position-independent (simplified)
        std::memcpy(v_data[i].data(), concept_k[concept].data(), head_dim * sizeof(float));
    }

    // ---- Test A: V6 (no RoPE awareness) ----
    std::cout << "\n=== Test A: V6 (No RoPE Awareness) ===" << std::endl;

    // Create one V6 instance per head (simulating full model)
    // For this test, we use a single head to show the effect clearly
    KVVectorcompV7 vcomp_v6(head_dim, ltm_slots, stm_size);  // rope_enabled=false by default

    std::vector<uint32_t> v6_ids(n_tokens);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_tokens; ++i) {
        v6_ids[i] = vcomp_v6.encode_shim(k_rotated[i].data(), v_data[i].data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    // Decode and compare
    std::vector<float> v6_decoded_k(head_dim);
    float v6_k_sim_total = 0.0f, v6_k_sim_min = 1.0f;
    float v6_v_sim_total = 0.0f, v6_v_sim_min = 1.0f;
    int v6_checks = 0;
    for (int i = 0; i < n_tokens; ++i) {
        try {
            vcomp_v6.decode_shim(&v6_ids[i], 1, v6_decoded_k.data(), v6_decoded_k.data());
            float k_sim = compute_cosine_similarity(k_rotated[i].data(), v6_decoded_k.data(), head_dim);
            float v_sim = compute_cosine_similarity(v_data[i].data(), v6_decoded_k.data(), head_dim);
            v6_k_sim_total += k_sim;
            v6_v_sim_total += v_sim;
            if (k_sim < v6_k_sim_min) v6_k_sim_min = k_sim;
            if (v_sim < v6_v_sim_min) v6_v_sim_min = v_sim;
            v6_checks++;
        } catch (...) {}
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "  Encode time: " << std::fixed << std::setprecision(1)
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;
    std::cout << "  Decode time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms" << std::endl;
    std::cout << "  Active slots: " << vcomp_v6.get_active_cb_count() << "/" << ltm_slots << std::endl;
    std::cout << "  K similarity: avg=" << std::fixed << std::setprecision(4)
              << (v6_checks > 0 ? v6_k_sim_total / v6_checks : 0.0f)
              << " min=" << v6_k_sim_min << std::endl;
    std::cout << "  V similarity: avg=" << std::fixed << std::setprecision(4)
              << (v6_checks > 0 ? v6_v_sim_total / v6_checks : 0.0f)
              << " min=" << v6_v_sim_min << std::endl;

    // ---- Test B: V7 (RoPE-Aware) ----
    std::cout << "\n=== Test B: V7 (RoPE-Aware, freq_base=" << rope_freq_base << ") ===" << std::endl;

    KVVectorcompV7 vcomp_v7(head_dim, ltm_slots, stm_size,
                             rope_freq_base, 0.98f, 0.92f, 0.85f, 0.1f);

    std::vector<uint32_t> v7_ids(n_tokens);
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_tokens; ++i) {
        v7_ids[i] = vcomp_v7.encode_shim_rope(k_rotated[i].data(), v_data[i].data(), i);
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    // Decode and compare
    // Note: V7 stores derotated (content) K in the codebook.
    // To compare with original, we need to re-rotate the decoded K.
    std::vector<float> v7_decoded_k(head_dim);
    std::vector<float> v7_rerotated_k(head_dim);
    float v7_k_sim_total = 0.0f, v7_k_sim_min = 1.0f;
    float v7_v_sim_total = 0.0f, v7_v_sim_min = 1.0f;
    int v7_checks = 0;
    int v7_evicted = 0;

    for (int i = 0; i < n_tokens; ++i) {
        try {
            vcomp_v7.decode_shim(&v7_ids[i], 1, v7_decoded_k.data(), v7_decoded_k.data());
            // Re-rotate the decoded content K to compare with original rotated K
            apply_rope(v7_decoded_k.data(), i, v7_rerotated_k.data(), head_dim, rope_freq_base);

            float k_sim = compute_cosine_similarity(k_rotated[i].data(), v7_rerotated_k.data(), head_dim);
            float v_sim = compute_cosine_similarity(v_data[i].data(), v7_decoded_k.data(), head_dim);
            v7_k_sim_total += k_sim;
            v7_v_sim_total += v_sim;
            if (k_sim < v7_k_sim_min) v7_k_sim_min = k_sim;
            if (v_sim < v7_v_sim_min) v7_v_sim_min = v_sim;
            v7_checks++;
        } catch (...) {
            v7_evicted++;
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();

    std::cout << "  Encode time: " << std::fixed << std::setprecision(1)
              << std::chrono::duration<double, std::milli>(t4 - t3).count() << " ms" << std::endl;
    std::cout << "  Decode time: " << std::chrono::duration<double, std::milli>(t5 - t4).count() << " ms" << std::endl;
    std::cout << "  Active slots: " << vcomp_v7.get_active_cb_count() << "/" << ltm_slots << std::endl;
    std::cout << "  Evicted: " << v7_evicted << std::endl;
    std::cout << "  K similarity: avg=" << std::fixed << std::setprecision(4)
              << (v7_checks > 0 ? v7_k_sim_total / v7_checks : 0.0f)
              << " min=" << v7_k_sim_min << std::endl;
    std::cout << "  V similarity: avg=" << std::fixed << std::setprecision(4)
              << (v7_checks > 0 ? v7_v_sim_total / v7_checks : 0.0f)
              << " min=" << v7_v_sim_min << std::endl;

    // ---- Test C: Multi-head simulation ----
    std::cout << "\n=== Test C: Multi-Head Simulation (32 heads x 24 layers) ===" << std::endl;

    int total_vectors = n_tokens * n_heads * n_layers;
    std::cout << "  Total vectors: " << total_vectors << std::endl;

    // V6 baseline
    float v6_multi_k_total = 0.0f, v6_multi_k_min = 1.0f;
    float v6_multi_v_total = 0.0f, v6_multi_v_min = 1.0f;
    int v6_multi_checks = 0;

    // V7 RoPE-aware
    float v7_multi_k_total = 0.0f, v7_multi_k_min = 1.0f;
    float v7_multi_v_total = 0.0f, v7_multi_v_min = 1.0f;
    int v7_multi_checks = 0;
    int v7_multi_evicted = 0;

    auto t6 = std::chrono::high_resolution_clock::now();

    for (int l = 0; l < n_layers; ++l) {
        // Each layer has slightly different content vectors
        std::vector<std::vector<float>> layer_concept_k(n_concepts, std::vector<float>(head_dim));
        std::mt19937 layer_gen(static_cast<uint32_t>(l * 1000 + 42));
        std::normal_distribution<float> layer_dist(0.0f, 1.0f / std::sqrt(static_cast<float>(head_dim)));
        for (int c = 0; c < n_concepts; ++c) {
            for (int d = 0; d < head_dim; ++d) {
                layer_concept_k[c][d] = layer_dist(layer_gen);
            }
            float norm = 0.0f;
            for (float f : layer_concept_k[c]) norm += f * f;
            norm = std::sqrt(norm);
            if (norm > 1e-9f) for (float& f : layer_concept_k[c]) f /= norm;
        }

        KVVectorcompV7 v6_layer(head_dim, ltm_slots, stm_size);
        KVVectorcompV7 v7_layer(head_dim, ltm_slots, stm_size,
                                 rope_freq_base, 0.98f, 0.92f, 0.85f, 0.1f);

        for (int h = 0; h < n_heads; ++h) {
            // Each head has slightly different projections
            std::vector<std::vector<float>> head_k_rotated(n_tokens, std::vector<float>(head_dim));
            std::vector<std::vector<float>> head_v(n_tokens, std::vector<float>(head_dim));

            for (int i = 0; i < n_tokens; ++i) {
                int concept = token_concepts[i];
                // Add small head-specific variation
                std::vector<float> head_content(head_dim);
                for (int d = 0; d < head_dim; ++d) {
                    head_content[d] = layer_concept_k[concept][d] + layer_dist(layer_gen) * 0.01f;
                }
                float norm = 0.0f;
                for (float f : head_content) norm += f * f;
                norm = std::sqrt(norm);
                if (norm > 1e-9f) for (float& f : head_content) f /= norm;

                apply_rope(head_content.data(), i, head_k_rotated[i].data(), head_dim, rope_freq_base);
                std::memcpy(head_v[i].data(), head_content.data(), head_dim * sizeof(float));
            }

            // V6 encode
            std::vector<uint32_t> v6_layer_ids(n_tokens);
            for (int i = 0; i < n_tokens; ++i) {
                v6_layer_ids[i] = v6_layer.encode_shim(head_k_rotated[i].data(), head_v[i].data());
            }

            // V7 encode
            std::vector<uint32_t> v7_layer_ids(n_tokens);
            for (int i = 0; i < n_tokens; ++i) {
                v7_layer_ids[i] = v7_layer.encode_shim_rope(head_k_rotated[i].data(), head_v[i].data(), i);
            }

            // V6 decode & compare
            std::vector<float> dec_k(head_dim);
            for (int i = 0; i < n_tokens; ++i) {
                try {
                    v6_layer.decode_shim(&v6_layer_ids[i], 1, dec_k.data(), dec_k.data());
                    float k_sim = compute_cosine_similarity(head_k_rotated[i].data(), dec_k.data(), head_dim);
                    float v_sim = compute_cosine_similarity(head_v[i].data(), dec_k.data(), head_dim);
                    v6_multi_k_total += k_sim;
                    v6_multi_v_total += v_sim;
                    if (k_sim < v6_multi_k_min) v6_multi_k_min = k_sim;
                    if (v_sim < v6_multi_v_min) v6_multi_v_min = v_sim;
                    v6_multi_checks++;
                } catch (...) {}
            }

            // V7 decode & compare (re-rotate)
            std::vector<float> dec_k_content(head_dim);
            std::vector<float> dec_k_rerotated(head_dim);
            for (int i = 0; i < n_tokens; ++i) {
                try {
                    v7_layer.decode_shim(&v7_layer_ids[i], 1, dec_k_content.data(), dec_k_content.data());
                    apply_rope(dec_k_content.data(), i, dec_k_rerotated.data(), head_dim, rope_freq_base);
                    float k_sim = compute_cosine_similarity(head_k_rotated[i].data(), dec_k_rerotated.data(), head_dim);
                    float v_sim = compute_cosine_similarity(head_v[i].data(), dec_k_content.data(), head_dim);
                    v7_multi_k_total += k_sim;
                    v7_multi_v_total += v_sim;
                    if (k_sim < v7_multi_k_min) v7_multi_k_min = k_sim;
                    if (v_sim < v7_multi_v_min) v7_multi_v_min = v_sim;
                    v7_multi_checks++;
                } catch (...) {
                    v7_multi_evicted++;
                }
            }
        }
    }

    auto t7 = std::chrono::high_resolution_clock::now();
    double multi_ms = std::chrono::duration<double, std::milli>(t7 - t6).count();

    std::cout << "  Total time: " << std::fixed << std::setprecision(0) << multi_ms << " ms" << std::endl;
    std::cout << "  Vectors/sec: " << std::fixed << std::setprecision(0)
              << (total_vectors / (multi_ms / 1000.0)) << std::endl;
    std::cout << "\n  V6 (no RoPE awareness):" << std::endl;
    std::cout << "    K similarity: avg=" << std::fixed << std::setprecision(4)
              << (v6_multi_checks > 0 ? v6_multi_k_total / v6_multi_checks : 0.0f)
              << " min=" << v6_multi_k_min << std::endl;
    std::cout << "    V similarity: avg=" << std::fixed << std::setprecision(4)
              << (v6_multi_checks > 0 ? v6_multi_v_total / v6_multi_checks : 0.0f)
              << " min=" << v6_multi_v_min << std::endl;
    std::cout << "\n  V7 (RoPE-aware):" << std::endl;
    std::cout << "    K similarity: avg=" << std::fixed << std::setprecision(4)
              << (v7_multi_checks > 0 ? v7_multi_k_total / v7_multi_checks : 0.0f)
              << " min=" << v7_multi_k_min << std::endl;
    std::cout << "    V similarity: avg=" << std::fixed << std::setprecision(4)
              << (v7_multi_checks > 0 ? v7_multi_v_total / v7_multi_checks : 0.0f)
              << " min=" << v7_multi_v_min << std::endl;
    std::cout << "    Evicted: " << v7_multi_evicted << std::endl;

    std::cout << "\n============================================================" << std::endl;
    std::cout << "  V7 RoPE-Aware Test Complete!" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
