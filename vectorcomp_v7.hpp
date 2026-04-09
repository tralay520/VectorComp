// Vectorcomp V7 — RoPE-Aware KV Compression
// Copyright (C) 2026 Tracy
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.


#ifndef VECTORCOMP_HPP
#define VECTORCOMP_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <cstddef>
#include <mutex>

class KVVectorcompV7 {
    static constexpr uint32_t RAW_FLAG = 1u << 31;

    std::vector<float> k_codebook, v_codebook;
    std::vector<float> k_raw_buffer, v_raw_buffer;
    struct Metadata { bool is_active = false; uint32_t usage_count = 0; uint64_t last_step = 0; };
    std::vector<Metadata> cb_metadata;

    int head_dim;
    int max_cb_slots;
    int max_raw_size;
    float high_strict, high_loose, medium_thresh;
    float learning_rate;

    // RoPE parameters for de-rotation before clustering
    float rope_freq_base;
    bool rope_enabled;
    std::vector<float> rope_inv_freq;  // Pre-computed 1/theta for each dimension pair

    uint64_t global_step = 0, raw_ptr = 0, raw_total_count = 0;

    mutable std::mutex mtx;

    float compute_cosine_similarity(const float* a, const float* b) const;
    int find_best_ltm_match(const float* k_new, float& out_sim) const;
    int get_evictable_ltm_slot() const;
    void update_centroid(int slot_idx, const float* k_new, const float* v_new);

    // Strip RoPE rotation from a K vector to get the content-only representation
    void derotate_k(const float* k_rotated, int position, float* k_content) const;

public:
    // V6-compatible constructor (no RoPE)
    KVVectorcompV7(int head_dim, int max_cb_slots = 512, int max_raw_size = 1024,
                   float high_strict = 0.98f, float high_loose = 0.92f, float medium_thresh = 0.85f,
                   float learning_rate = 0.1f)
        : head_dim(head_dim), max_cb_slots(max_cb_slots), max_raw_size(max_raw_size),
          high_strict(high_strict), high_loose(high_loose), medium_thresh(medium_thresh),
          learning_rate(learning_rate),
          rope_freq_base(10000.0f), rope_enabled(false) {

        if (medium_thresh > high_loose || high_loose > high_strict) {
            throw std::invalid_argument(
                "Threshold ordering violated: medium_thresh <= high_loose <= high_strict required");
        }
        if (head_dim <= 0 || max_cb_slots <= 0 || max_raw_size <= 0) {
            throw std::invalid_argument("head_dim, max_cb_slots, and max_raw_size must be positive");
        }

        k_codebook.resize(static_cast<size_t>(max_cb_slots) * head_dim, 0.0f);
        v_codebook.resize(static_cast<size_t>(max_cb_slots) * head_dim, 0.0f);
        k_raw_buffer.resize(static_cast<size_t>(max_raw_size) * head_dim, 0.0f);
        v_raw_buffer.resize(static_cast<size_t>(max_raw_size) * head_dim, 0.0f);
        cb_metadata.resize(max_cb_slots);
    }

    // V7 constructor with RoPE awareness
    KVVectorcompV7(int head_dim, int max_cb_slots, int max_raw_size,
                   float rope_freq_base, float high_strict, float high_loose,
                   float medium_thresh, float learning_rate)
        : head_dim(head_dim), max_cb_slots(max_cb_slots), max_raw_size(max_raw_size),
          high_strict(high_strict), high_loose(high_loose), medium_thresh(medium_thresh),
          learning_rate(learning_rate),
          rope_freq_base(rope_freq_base), rope_enabled(true) {

        if (medium_thresh > high_loose || high_loose > high_strict) {
            throw std::invalid_argument(
                "Threshold ordering violated: medium_thresh <= high_loose <= high_strict required");
        }
        if (head_dim <= 0 || max_cb_slots <= 0 || max_raw_size <= 0) {
            throw std::invalid_argument("head_dim, max_cb_slots, and max_raw_size must be positive");
        }

        // Pre-compute inverse frequencies: theta_i = base^(2i/d)
        rope_inv_freq.resize(head_dim / 2);
        for (int i = 0; i < head_dim / 2; ++i) {
            rope_inv_freq[i] = 1.0f / std::pow(rope_freq_base, static_cast<float>(2 * i) / head_dim);
        }

        k_codebook.resize(static_cast<size_t>(max_cb_slots) * head_dim, 0.0f);
        v_codebook.resize(static_cast<size_t>(max_cb_slots) * head_dim, 0.0f);
        k_raw_buffer.resize(static_cast<size_t>(max_raw_size) * head_dim, 0.0f);
        v_raw_buffer.resize(static_cast<size_t>(max_raw_size) * head_dim, 0.0f);
        cb_metadata.resize(max_cb_slots);
    }

    KVVectorcompV7(const KVVectorcompV7&) = delete;
    KVVectorcompV7& operator=(const KVVectorcompV7&) = delete;
    KVVectorcompV7(KVVectorcompV7&&) noexcept = default;
    KVVectorcompV7& operator=(KVVectorcompV7&&) noexcept = default;

    // V6 API: encode without position info (no RoPE stripping)
    uint32_t encode_shim(const float* k_new, const float* v_new);
    void decode_shim(const uint32_t* ids, int seq_len, float* out_k, float* out_v);

    // V7 API: encode with position info (strips RoPE before clustering)
    uint32_t encode_shim_rope(const float* k_rotated, const float* v_new, int position);

    size_t get_active_cb_count() const;
    uint64_t get_global_step() const { return global_step; }
};

#endif
