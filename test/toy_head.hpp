#ifndef TOY_HEAD_HPP
#define TOY_HEAD_HPP

#include <vector>
#include <cstdint>
#include <string>

// A minimal single-head transformer that produces KV vectors
// and feeds them through Vectorcomp compression.
// No GPU, no trained weights needed — purely CPU, purely simple.
class ToyHead {
    int vocab_size;    // size of our tiny vocabulary
    int embed_dim;     // embedding dimension
    int head_dim;      // KV projection dimension (what vectorcomp sees)

    // Learned-style weights (randomly initialized, fixed)
    std::vector<float> embed_table;    // [vocab_size, embed_dim]
    std::vector<float> W_k;            // [embed_dim, head_dim]
    std::vector<float> W_v;            // [embed_dim, head_dim]
    std::vector<float> W_o;            // [head_dim, embed_dim]

    // Pre-compute K and V for every token in vocab
    std::vector<float> token_k_cache;  // [vocab_size, head_dim]
    std::vector<float> token_v_cache;  // [vocab_size, head_dim]

    void matmul_vec(const std::vector<float>& W, int rows, int cols,
                    const float* x, float* out) const;

public:
    ToyHead(int vocab_size, int embed_dim, int head_dim, uint32_t seed = 42);

    // Get K, V for a single token (by token_id)
    const float* get_token_k(int token_id) const;
    const float* get_token_v(int token_id) const;

    // Run a full forward pass:
    //   1. Lookup embeddings for input token sequence
    //   2. Project to K, V
    //   3. Encode through vectorcomp -> compressed IDs
    //   4. Decode from vectorcomp -> reconstructed K, V
    //   5. Compute simple attention (dot-product) with reconstructed K
    //   6. Project V through output projection
    // Returns: reconstructed output embedding per token
    std::vector<float> forward_compressed(
        const std::vector<int>& token_ids,
        class KVVectorcompV6& vcomp,
        std::vector<uint32_t>& out_ids  // compressed IDs produced
    ) const;

    // Same as above but WITHOUT compression (baseline for comparison)
    std::vector<float> forward_uncompressed(
        const std::vector<int>& token_ids
    ) const;

    // Utility: compute cosine similarity between two flat vectors
    static float cosine_sim(const float* a, const float* b, int dim);

    // Utility: pretty-print token sequence (uses a simple internal vocab map)
    static std::string token_to_str(int token_id);
};

#endif
