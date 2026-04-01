#import <Metal/Metal.h>
#include "TransformerLayer.h"
#include <vector>
#include <algorithm>

void SparseTransformerLayer::init(int dim, int num_heads, int block_size, float density, int ffn_dim) {
    norm1.init(); norm2.init();
    attn = std::make_shared<SparseAttention>(dim, num_heads, block_size, density);
    ffn  = std::make_shared<SparseFFN>(dim, density, ffn_dim);
}

std::vector<std::shared_ptr<SparseTernaryLinear>> SparseTransformerLayer::all_sparse_weights() {
    auto aw = attn->get_internal_weights();
    auto fw = ffn->get_internal_weights();
    aw.insert(aw.end(), fw.begin(), fw.end());
    return aw;
}

Tier SparseTransformerLayer::dominant_tier(const TierManager& tm) const {
    int h=0, w=0, c=0;
    for (int i = 0; i < 7; ++i) {
        switch (tm.get_tier(hima_base_idx + i)) {
            case Tier::HOT:  h++; break;
            case Tier::WARM: w++; break;
            case Tier::COLD: c++; break;
        }
    }
    if (h >= w && h >= c) return Tier::HOT;
    if (w >= c) return Tier::WARM;
    return Tier::COLD;
}

void SparseTransformerLayer::clear_saved() {
    r_pre_attn.reset(); r_pre_ffn.reset();
    attn->clear_activations();
    ffn->clear_activations();
}
