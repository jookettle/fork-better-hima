#pragma once
#include <vector>
#include <memory>
#include "RMSNorm.h"
#include "SparseAttention.h"
#include "SparseFFN.h"
#include "TierManager.h"
#include "SparseTernaryLinear.h"

struct SparseTransformerLayer {
    RMSNorm norm1, norm2;
    std::shared_ptr<SparseAttention> attn;
    std::shared_ptr<SparseFFN> ffn;
    
    std::shared_ptr<Tensor> r_pre_attn, r_pre_ffn;
    int hima_base_idx = 0;

    void init(int dim, int num_heads, int block_size, float density, int ffn_dim);
    std::vector<std::shared_ptr<SparseTernaryLinear>> all_sparse_weights();
    Tier dominant_tier(const TierManager& tm) const;
    void clear_saved();
};
