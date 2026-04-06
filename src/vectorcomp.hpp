// Vectorcomp V6
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

class KVVectorcompV6 {
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

    uint64_t global_step = 0, raw_ptr = 0, raw_total_count = 0;

    mutable std::mutex mtx;

    float compute_cosine_similarity(const float* a, const float* b) const;
    int find_best_ltm_match(const float* k_new, float& out_sim) const;
    int get_evictable_ltm_slot() const;
    void update_centroid(int slot_idx, const float* k_new, const float* v_new);

public:
    KVVectorcompV6(int head_dim, int max_cb_slots = 512, int max_raw_size = 1024,
                   float high_strict = 0.98f, float high_loose = 0.92f, float medium_thresh = 0.85f,
                   float learning_rate = 0.1f)
        : head_dim(head_dim), max_cb_slots(max_cb_slots), max_raw_size(max_raw_size),
          high_strict(high_strict), high_loose(high_loose), medium_thresh(medium_thresh),
          learning_rate(learning_rate) {

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

    KVVectorcompV6(const KVVectorcompV6&) = delete;
    KVVectorcompV6& operator=(const KVVectorcompV6&) = delete;
    KVVectorcompV6(KVVectorcompV6&&) noexcept = default;
    KVVectorcompV6& operator=(KVVectorcompV6&&) noexcept = default;

    uint32_t encode_shim(const float* k_new, const float* v_new);
    void decode_shim(const uint32_t* ids, int seq_len, float* out_k, float* out_v);

    size_t get_active_cb_count() const;
    uint64_t get_global_step() const { return global_step; }
};

#endif
