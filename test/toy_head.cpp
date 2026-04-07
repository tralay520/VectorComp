#include "toy_head.hpp"
#include "vectorcomp.hpp"
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include <cstring>

ToyHead::ToyHead(int vocab_size, int embed_dim, int head_dim, uint32_t seed)
    : vocab_size(vocab_size), embed_dim(embed_dim), head_dim(head_dim) {

    embed_table.resize(static_cast<size_t>(vocab_size) * embed_dim);
    W_k.resize(static_cast<size_t>(embed_dim) * head_dim);
    W_v.resize(static_cast<size_t>(embed_dim) * head_dim);
    W_o.resize(static_cast<size_t>(head_dim) * embed_dim);
    token_k_cache.resize(static_cast<size_t>(vocab_size) * head_dim);
    token_v_cache.resize(static_cast<size_t>(vocab_size) * head_dim);

    std::mt19937 gen(seed);
    // Xavier-ish initialization
    float embed_std = 1.0f / std::sqrt(static_cast<float>(embed_dim));
    float proj_std  = 1.0f / std::sqrt(static_cast<float>(embed_dim + head_dim));

    std::normal_distribution<float> nd_embed(0.0f, embed_std);
    std::normal_distribution<float> nd_proj(0.0f, proj_std);

    for (auto& w : embed_table) w = nd_embed(gen);
    for (auto& w : W_k) w = nd_proj(gen);
    for (auto& w : W_v) w = nd_proj(gen);
    for (auto& w : W_o) w = nd_proj(gen);

    // Pre-compute K, V for every token
    for (int t = 0; t < vocab_size; ++t) {
        matmul_vec(W_k, embed_dim, head_dim,
                   &embed_table[static_cast<size_t>(t) * embed_dim],
                   &token_k_cache[static_cast<size_t>(t) * head_dim]);
        matmul_vec(W_v, embed_dim, head_dim,
                   &embed_table[static_cast<size_t>(t) * embed_dim],
                   &token_v_cache[static_cast<size_t>(t) * head_dim]);
    }
}

void ToyHead::matmul_vec(const std::vector<float>& W, int rows, int cols,
                         const float* x, float* out) const {
    std::memset(out, 0, static_cast<size_t>(cols) * sizeof(float));
    for (int r = 0; r < rows; ++r) {
        float val = x[r];
        for (int c = 0; c < cols; ++c) {
            out[c] += val * W[static_cast<size_t>(r) * cols + c];
        }
    }
}

const float* ToyHead::get_token_k(int token_id) const {
    return &token_k_cache[static_cast<size_t>(token_id) * head_dim];
}

const float* ToyHead::get_token_v(int token_id) const {
    return &token_v_cache[static_cast<size_t>(token_id) * head_dim];
}

std::vector<float> ToyHead::forward_compressed(
    const std::vector<int>& token_ids,
    KVVectorcompV6& vcomp,
    std::vector<uint32_t>& out_ids
) const {
    int seq_len = static_cast<int>(token_ids.size());
    out_ids.resize(seq_len);

    // Step 1: Encode each token's K, V through vectorcomp
    for (int i = 0; i < seq_len; ++i) {
        const float* k = get_token_k(token_ids[i]);
        const float* v = get_token_v(token_ids[i]);
        out_ids[i] = vcomp.encode_shim(k, v);
    }

    // Step 2: Decode all at once
    std::vector<float> decoded_k(seq_len * head_dim);
    std::vector<float> decoded_v(seq_len * head_dim);
    vcomp.decode_shim(out_ids.data(), seq_len, decoded_k.data(), decoded_v.data());

    // Step 3: Simple self-attention with reconstructed K
    // For each query position i, compute attention over all j <= i
    std::vector<float> output(seq_len * embed_dim, 0.0f);

    for (int i = 0; i < seq_len; ++i) {
        const float* q = get_token_k(token_ids[i]); // reuse K projection as Q

        // Compute attention scores
        std::vector<float> scores(i + 1);
        float max_score = -1e9f;
        for (int j = 0; j <= i; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q[d] * decoded_k[static_cast<size_t>(j) * head_dim + d];
            }
            scores[j] = dot / std::sqrt(static_cast<float>(head_dim));
            if (scores[j] > max_score) max_score = scores[j];
        }

        // Softmax
        float sum_exp = 0.0f;
        for (int j = 0; j <= i; ++j) {
            scores[j] = std::exp(scores[j] - max_score);
            sum_exp += scores[j];
        }
        for (int j = 0; j <= i; ++j) scores[j] /= sum_exp;

        // Weighted sum of V
        std::vector<float> v_sum(head_dim, 0.0f);
        for (int j = 0; j <= i; ++j) {
            for (int d = 0; d < head_dim; ++d) {
                v_sum[d] += scores[j] * decoded_v[static_cast<size_t>(j) * head_dim + d];
            }
        }

        // Output projection
        matmul_vec(W_o, head_dim, embed_dim, v_sum.data(),
                   &output[static_cast<size_t>(i) * embed_dim]);
    }

    return output;
}

std::vector<float> ToyHead::forward_uncompressed(
    const std::vector<int>& token_ids
) const {
    int seq_len = static_cast<int>(token_ids.size());
    std::vector<float> output(seq_len * embed_dim, 0.0f);

    for (int i = 0; i < seq_len; ++i) {
        const float* q = get_token_k(token_ids[i]);

        std::vector<float> scores(i + 1);
        float max_score = -1e9f;
        for (int j = 0; j <= i; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q[d] * get_token_k(token_ids[j])[d];
            }
            scores[j] = dot / std::sqrt(static_cast<float>(head_dim));
            if (scores[j] > max_score) max_score = scores[j];
        }

        float sum_exp = 0.0f;
        for (int j = 0; j <= i; ++j) {
            scores[j] = std::exp(scores[j] - max_score);
            sum_exp += scores[j];
        }
        for (int j = 0; j <= i; ++j) scores[j] /= sum_exp;

        std::vector<float> v_sum(head_dim, 0.0f);
        for (int j = 0; j <= i; ++j) {
            const float* v = get_token_v(token_ids[j]);
            for (int d = 0; d < head_dim; ++d) {
                v_sum[d] += scores[j] * v[d];
            }
        }

        matmul_vec(W_o, head_dim, embed_dim, v_sum.data(),
                   &output[static_cast<size_t>(i) * embed_dim]);
    }

    return output;
}

float ToyHead::cosine_sim(const float* a, const float* b, int dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-9f || nb < 1e-9f) return 0.0f;
    return dot / std::sqrt(na * nb);
}

std::string ToyHead::token_to_str(int token_id) {
    static const char* vocab[] = {
        "<pad>", "the", "a", "is", "it", "of", "and", "to", "in", "that",
        "this", "was", "for", "on", "are", "with", "as", "at", "be", "this",
        "from", "have", "not", "but", "had", "what", "all", "were", "when", "we",
        "there", "can", "your", "which", "their", "if", "do", "will", "each", "about",
        "how", "up", "out", "them", "then", "she", "many", "some", "so", "these",
        "would", "other", "into", "has", "more", "her", "two", "like", "him", "see",
        "time", "could", "no", "make", "than", "been", "its", "who", "now", "my",
        "made", "after", "did", "just", "around", "name", "only", "new", "very", "people",
        "take", "come", "may", "back", "after", "over", "think", "also", "before", "use",
        "way", "look", "want", "give", "day", "most", "us", "good", "well", "long"
    };
    static const int vocab_count = sizeof(vocab) / sizeof(vocab[0]);
    if (token_id < 0 || token_id >= vocab_count) return "<unk>";
    return vocab[token_id];
}
