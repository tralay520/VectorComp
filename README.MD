Vectorcomp V7
Vectorcomp V7is a semantic KV‑cache compression system designed to reduce memory footprint while increasing effective long‑term memory capacity for transformer models. It uses a hybrid LTM/STM architecture with centroid drift, strict reuse, and eviction‑safe sliding‑window behavior.

Features
Lossless STM round‑trip

Stable LTM clustering with controlled centroid drift

Strict match preservation

Sliding‑window STM eviction safety

Increased semantic memory density

Fully tested (12/12 functional + stress tests)

Header‑only API surface + single C++ implementation file

Quick Start
bash
# Build
g++ -std=c++17 vectorcomp.cpp test_vectorcomp.cpp -o vcomp_test

# Run tests
./vcomp_test
How It Works
On every encode_shim(k, v) call, the incoming key vector is compared against the LTM codebook using cosine similarity:

Similarity Range	Action	ID Type
≥ 0.98 (high_strict)	LTM match, return codebook index, no drift	LTM
≥ 0.92 (high_loose)	LTM match, return index, update centroid	LTM
≥ 0.85 (medium_thresh)	Store in STM ring buffer	RAW
New LTM slot (evict LRU if full)	LTM
The returned 32‑bit ID packs everything:

bit 31 → raw/STM flag

bits 0–30 → index into LTM or STM

Usage
cpp
#include "vectorcomp.hpp"

KVVectorcompV6 vcomp(
    /* head_dim */       128,
    /* max_cb_slots */   512,
    /* max_raw_size */   1024
);

// Encode: returns a 32-bit ID
uint32_t id = vcomp.encode_shim(k_vec, v_vec);

// Decode: reconstructs K and V from the ID
float k_out[128], v_out[128];
vcomp.decode_shim(&id, 1, k_out, v_out);
Test Suite
Vectorcomp V6 includes a comprehensive 12‑test validation suite:

#	Test Name	What It Verifies
1	Basic LTM Insertion & Reuse	             Unique storage, strict ID reuse
2	STM Insertion & Decode	                 Perturbed vectors round‑trip correctly
3	STM Eviction	                         Ring buffer overflow throws on stale IDs
4	LTM Eviction	                         LRU‑score eviction when slots are full
5	Centroid Drift	                         Controlled drift on medium‑high matches
6	High Strict No Drift	                 Exact matches preserve vectors perfectly
7	Out‑of‑Range Decode	                     Invalid IDs are rejected
8	Sequence Decode	                         Multi‑token batch decode
9	Global Step Counter	                     Step tracking for LRU scoring
10	Jitter Stability	                     Gaussian noise doesn’t cause centroid chaos
11	Goldfish Memory	                         100 concepts survive 1000 junk tokens
12	Memory Profiling	                     RAM footprint vs raw KV cache
All tests pass.

Why Vectorcomp Exists
Modern transformer models waste enormous amounts of memory storing redundant KV vectors. Vectorcomp V6 compresses repeated patterns semantically, allowing models to retain more meaningful context without increasing memory usage.

This enables:

longer effective context

reduced RAM footprint

more stable character and topic memory

better inference efficiency

License
Vectorcomp V6 is licensed under the GNU GPLv3.
This ensures the project remains free and open, and prevents proprietary forks or closed‑source commercialization.

Contributing
See CONTRIBUTING.md for:

build instructions

code style

test requirements

how to submit PRs

Author
Created by Tracy (2026)
