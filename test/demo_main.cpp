#include "../src/toy_head.hpp"
#include "../src/vectorcomp.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <map>
#include <iomanip>
#include <random>

// Realistic transformer-head benchmark for Vectorcomp V6
// Simulates context-dependent KV variation (like a real model)
// Runs in seconds on any CPU — no GPU needed

class SimpleTokenizer {
    std::map<std::string, int> word_to_id;
public:
    SimpleTokenizer() {
        for (int i = 0; i < 100; ++i) {
            word_to_id[ToyHead::token_to_str(i)] = i;
        }
    }

    std::vector<int> tokenize(const std::string& text) {
        std::vector<int> tokens;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            std::string cleaned;
            for (char c : word) {
                if (std::isalpha(c) || c == '\'') cleaned += std::tolower(c);
            }
            if (cleaned.empty()) continue;
            auto it = word_to_id.find(cleaned);
            tokens.push_back(it != word_to_id.end() ? it->second : 0);
        }
        return tokens;
    }
};

// Simulates a transformer head where the same token produces slightly
// different KV vectors depending on position/context (like a real model)
std::vector<float> add_context_noise(const float* base, int dim,
                                      float sigma, int seed) {
    std::vector<float> result(dim);
    std::mt19937 gen(seed);
    std::normal_distribution<float> noise(0.0f, sigma);
    for (int i = 0; i < dim; ++i) {
        result[i] = base[i] + noise(gen);
    }
    // Normalize to unit vector
    float norm = 0.0f;
    for (float f : result) norm += f * f;
    norm = std::sqrt(norm);
    if (norm > 1e-9f) for (float& f : result) f /= norm;
    return result;
}

void run_realistic_benchmark(const std::string& name,
                              const std::vector<int>& tokens,
                              ToyHead& head, int head_dim, int embed_dim,
                              int ltm_slots, int stm_size,
                              float context_sigma) {
    std::cout << "\n--- " << name << " ---" << std::endl;
    std::cout << "  Tokens: " << tokens.size()
              << " (context noise σ=" << std::fixed << std::setprecision(3)
              << context_sigma << ")" << std::endl;

    // Build context-dependent KV pairs
    std::vector<std::vector<float>> ks, vs;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const float* base_k = head.get_token_k(tokens[i]);
        const float* base_v = head.get_token_v(tokens[i]);
        ks.push_back(add_context_noise(base_k, head_dim, context_sigma, static_cast<int>(i * 1000 + 7)));
        vs.push_back(add_context_noise(base_v, head_dim, context_sigma, static_cast<int>(i * 1000 + 13)));
    }

    // Uncompressed baseline (uses exact token KVs, no noise)
    auto t0 = std::chrono::high_resolution_clock::now();
    auto output_raw = head.forward_uncompressed(tokens);
    auto t1 = std::chrono::high_resolution_clock::now();
    double raw_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Compressed with context-dependent KVs
    KVVectorcompV6 vcomp(head_dim, ltm_slots, stm_size);
    std::vector<uint32_t> compressed_ids(tokens.size());
    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < tokens.size(); ++i) {
        compressed_ids[i] = vcomp.encode_shim(ks[i].data(), vs[i].data());
    }
    // Decode one at a time (catch evicted STM entries)
    std::vector<float> decoded_k(tokens.size() * head_dim);
    std::vector<float> decoded_v(tokens.size() * head_dim);
    int evicted = 0;
    auto t3 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < tokens.size(); ++i) {
        try {
            vcomp.decode_shim(&compressed_ids[i], 1,
                              &decoded_k[i * head_dim], &decoded_v[i * head_dim]);
        } catch (const std::runtime_error&) {
            evicted++;
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double comp_ms = std::chrono::duration<double, std::milli>(t4 - t2).count();

    // Stats
    int ltm = 0, stm = 0;
    for (uint32_t id : compressed_ids) {
        if ((id >> 31) & 1) stm++;
        else ltm++;
    }
    std::vector<uint32_t> unique_ids = compressed_ids;
    std::sort(unique_ids.begin(), unique_ids.end());
    unique_ids.erase(std::unique(unique_ids.begin(), unique_ids.end()), unique_ids.end());

    float raw_kv_bytes = static_cast<float>(tokens.size()) * 2 * head_dim * sizeof(float);
    float compressed_bytes = static_cast<float>(tokens.size()) * sizeof(uint32_t);

    std::cout << "  LTM slots: " << vcomp.get_active_cb_count() << "/" << ltm_slots
              << " | LTM hits: " << ltm << " | STM writes: " << stm << std::endl;
    std::cout << "  Unique IDs: " << unique_ids.size() << "/" << tokens.size()
              << " | Compression: " << std::fixed << std::setprecision(1)
              << (raw_kv_bytes / compressed_bytes) << "x" << std::endl;

    // Quality: compare decoded KVs vs what was stored
    float kv_sim_total = 0.0f, kv_sim_min = 1.0f;
    for (size_t i = 0; i < tokens.size(); ++i) {
        float k_sim = ToyHead::cosine_sim(ks[i].data(), &decoded_k[i * head_dim], head_dim);
        float v_sim = ToyHead::cosine_sim(vs[i].data(), &decoded_v[i * head_dim], head_dim);
        float avg = (k_sim + v_sim) / 2.0f;
        kv_sim_total += avg;
        if (avg < kv_sim_min) kv_sim_min = avg;
    }
    std::cout << "  KV decode similarity: avg=" << std::fixed << std::setprecision(4)
              << (kv_sim_total / (tokens.size() - evicted))
              << " min=" << std::fixed << std::setprecision(4) << kv_sim_min
              << " (evicted: " << evicted << ")" << std::endl;
    std::cout << "  Time: raw=" << std::fixed << std::setprecision(2) << raw_ms
              << "ms comp=" << comp_ms << "ms" << std::endl;
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "  Vectorcomp V6 — Realistic Transformer Head Benchmark" << std::endl;
    std::cout << "============================================================" << std::endl;

    int vocab_size = 100, embed_dim = 64, head_dim = 32;
    std::cout << "\nConfig: vocab=" << vocab_size << ", embed=" << embed_dim
              << ", head_dim=" << head_dim << std::endl;

    ToyHead head(vocab_size, embed_dim, head_dim, 42);
    SimpleTokenizer tokenizer;

    // Test 1: Exact repetition, tiny noise (mostly LTM hits)
    {
        std::string text = "the cat is on the mat the cat is on the mat "
                          "the dog is in the yard the dog is in the yard";
        auto tokens = tokenizer.tokenize(text);
        run_realistic_benchmark("Test 1: Repetition (σ=0.005, mostly LTM)",
                                tokens, head, head_dim, embed_dim, 32, 64, 0.005f);
    }

    // Test 2: Natural prose, moderate noise (LTM + some STM)
    {
        std::string text = "the cat and the dog are good friends the cat is small "
                          "and the dog is big but the cat and the dog play together "
                          "the cat is fast and the dog is slow but the cat and the dog "
                          "are friends and the cat is happy";
        auto tokens = tokenizer.tokenize(text);
        run_realistic_benchmark("Test 2: Natural Prose (σ=0.02, LTM+STM)",
                                tokens, head, head_dim, embed_dim, 16, 32, 0.02f);
    }

    // Test 3: Long sequence, higher noise (stress test STM ring buffer)
    {
        std::string text = "the cat is on the mat and the dog is in the yard "
                          "the cat is happy and the dog is happy too "
                          "the cat and the dog are friends and the cat is good "
                          "and the dog is good and the cat is fast and the dog is slow "
                          "the cat is small and the dog is big but the cat and the dog "
                          "are good friends and the cat is on the mat and the dog is in "
                          "the yard and the cat is happy and the dog is happy too";
        auto tokens = tokenizer.tokenize(text);
        run_realistic_benchmark("Test 3: Long Sequence (σ=0.05, STM stress)",
                                tokens, head, head_dim, embed_dim, 16, 16, 0.05f);
    }

    // Test 4: Low redundancy, high noise (worst case)
    {
        std::string text = "the cat is new and the dog was old but the cat can run "
                          "and the dog will walk so the cat may jump and the dog had "
                          "to look at what the cat could do and the dog did think about "
                          "how the cat would act and the cat did come back after the dog";
        auto tokens = tokenizer.tokenize(text);
        run_realistic_benchmark("Test 4: Low Redundancy (σ=0.08, worst case)",
                                tokens, head, head_dim, embed_dim, 32, 32, 0.08f);
    }

    // Test 5: Memory profiling
    {
        std::cout << "\n--- Test 5: Memory Profiling (200 tokens) ---" << std::endl;
        int num_tokens = 200;
        size_t raw_kv = static_cast<size_t>(num_tokens) * 2 * head_dim * sizeof(float);
        size_t vcomp_cb = 64 * head_dim * 2 * sizeof(float);
        size_t vcomp_raw = 128 * head_dim * 2 * sizeof(float);
        size_t vcomp_meta = 64 * 16;
        size_t vcomp_total = vcomp_cb + vcomp_raw + vcomp_meta;
        size_t comp_ids = static_cast<size_t>(num_tokens) * sizeof(uint32_t);

        std::cout << "  Raw KV cache:    " << (raw_kv / 1024.0) << " KB" << std::endl;
        std::cout << "  Vectorcomp:      " << (vcomp_total / 1024.0) << " KB" << std::endl;
        std::cout << "  Compressed IDs:  " << (comp_ids / 1024.0) << " KB" << std::endl;
        std::cout << "  Savings vs raw:  " << std::fixed << std::setprecision(1)
                  << ((1.0 - static_cast<double>(vcomp_total) / raw_kv) * 100.0) << "%" << std::endl;
        std::cout << "  ID compression:  " << std::fixed << std::setprecision(1)
                  << (static_cast<double>(raw_kv) / comp_ids) << "x" << std::endl;
    }

    std::cout << "\n============================================================" << std::endl;
    std::cout << "  Benchmark complete!" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
