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


#include "vectorcomp.hpp"
#include <algorithm>

float KVVectorcompV6::compute_cosine_similarity(const float* a, const float* b) const {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a < 1e-9f || norm_b < 1e-9f) return 0.0f;
    return dot / std::sqrt(norm_a * norm_b);
}

int KVVectorcompV6::find_best_ltm_match(const float* k_new, float& out_sim) const {
    int best_idx = -1;
    out_sim = -1.0f;
    for (int i = 0; i < max_cb_slots; ++i) {
        if (!cb_metadata[i].is_active) continue;
        float sim = compute_cosine_similarity(k_new, &k_codebook[static_cast<size_t>(i) * head_dim]);
        if (sim > out_sim) {
            out_sim = sim;
            best_idx = i;
        }
    }
    return best_idx;
}

void KVVectorcompV6::update_centroid(int idx, const float* k_new, const float* v_new) {
    float* k_slot = &k_codebook[static_cast<size_t>(idx) * head_dim];
    float* v_slot = &v_codebook[static_cast<size_t>(idx) * head_dim];
    for (int i = 0; i < head_dim; ++i) {
        k_slot[i] += learning_rate * (k_new[i] - k_slot[i]);
        v_slot[i] += learning_rate * (v_new[i] - v_slot[i]);
    }
}

int KVVectorcompV6::get_evictable_ltm_slot() const {
    for (int i = 0; i < max_cb_slots; ++i) {
        if (!cb_metadata[i].is_active) return i;
    }
    int best_victim = 0;
    float min_score = 1e20f;
    for (int i = 0; i < max_cb_slots; ++i) {
        uint64_t age = global_step - cb_metadata[i].last_step;
        float score = static_cast<float>(cb_metadata[i].usage_count) - (0.01f * static_cast<float>(age));
        if (score < min_score) {
            min_score = score;
            best_victim = i;
        }
    }
    return best_victim;
}

uint32_t KVVectorcompV6::encode_shim(const float* k_new, const float* v_new) {
    std::lock_guard<std::mutex> lock(mtx);

    global_step++;
    float best_sim = 0.0f;
    int best_idx = find_best_ltm_match(k_new, best_sim);

    if (best_idx != -1 && best_sim >= high_loose) {
        if (best_sim < high_strict) update_centroid(best_idx, k_new, v_new);
        cb_metadata[best_idx].usage_count++;
        cb_metadata[best_idx].last_step = global_step;
        return static_cast<uint32_t>(best_idx);
    }

    if (best_idx != -1 && best_sim >= medium_thresh) {
        auto raw_offset = static_cast<size_t>(raw_ptr) * head_dim;
        std::memcpy(&k_raw_buffer[raw_offset], k_new, static_cast<size_t>(head_dim) * sizeof(float));
        std::memcpy(&v_raw_buffer[raw_offset], v_new, static_cast<size_t>(head_dim) * sizeof(float));
        uint32_t id = RAW_FLAG | static_cast<uint32_t>(raw_total_count & 0x7FFFFFFF);
        raw_ptr = (raw_ptr + 1) % static_cast<uint64_t>(max_raw_size);
        raw_total_count++;
        return id;
    }

    int slot = get_evictable_ltm_slot();
    auto slot_offset = static_cast<size_t>(slot) * head_dim;
    std::memcpy(&k_codebook[slot_offset], k_new, static_cast<size_t>(head_dim) * sizeof(float));
    std::memcpy(&v_codebook[slot_offset], v_new, static_cast<size_t>(head_dim) * sizeof(float));
    cb_metadata[slot].is_active = true;
    cb_metadata[slot].usage_count = 1;
    cb_metadata[slot].last_step = global_step;
    return static_cast<uint32_t>(slot);
}

void KVVectorcompV6::decode_shim(const uint32_t* ids, int seq_len, float* out_k, float* out_v) {
    std::lock_guard<std::mutex> lock(mtx);

    for (int i = 0; i < seq_len; ++i) {
        bool is_raw = (ids[i] >> 31) & 1;
        uint32_t val = ids[i] & 0x7FFFFFFF;

        if (is_raw) {
            if (val >= raw_total_count) {
                throw std::runtime_error("STM Evicted: Accessing data outside sliding window.");
            }
            uint64_t oldest_valid = (raw_total_count > static_cast<uint64_t>(max_raw_size))
                ? raw_total_count - static_cast<uint64_t>(max_raw_size)
                : 0;
            if (static_cast<uint64_t>(val) < oldest_valid) {
                throw std::runtime_error("STM Evicted: Accessing data outside sliding window.");
            }
            int ring_idx = static_cast<int>(static_cast<uint64_t>(val) % static_cast<uint64_t>(max_raw_size));
            auto ring_offset = static_cast<size_t>(ring_idx) * head_dim;
            std::memcpy(out_k + static_cast<size_t>(i) * head_dim, &k_raw_buffer[ring_offset], static_cast<size_t>(head_dim) * sizeof(float));
            std::memcpy(out_v + static_cast<size_t>(i) * head_dim, &v_raw_buffer[ring_offset], static_cast<size_t>(head_dim) * sizeof(float));
        } else {
            if (val >= static_cast<uint32_t>(max_cb_slots)) {
                throw std::out_of_range("Codebook index out of range in decode_shim.");
            }
            auto cb_offset = static_cast<size_t>(val) * head_dim;
            std::memcpy(out_k + static_cast<size_t>(i) * head_dim, &k_codebook[cb_offset], static_cast<size_t>(head_dim) * sizeof(float));
            std::memcpy(out_v + static_cast<size_t>(i) * head_dim, &v_codebook[cb_offset], static_cast<size_t>(head_dim) * sizeof(float));
        }
    }
}

size_t KVVectorcompV6::get_active_cb_count() const {
    size_t count = 0;
    for (const auto& m : cb_metadata) if (m.is_active) count++;
    return count;
}
