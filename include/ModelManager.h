#pragma once
#include <string>
#include <vector>
#include "EmbeddingHead.h"
#include "TransformerLayer.h"

void save_full_model(
    const std::string& path,
    const FactoredEmbLMHead& emb_lm,
    const std::vector<SparseTransformerLayer>& tl,
    int dim, int layers, int heads, int vocab, float density);

bool load_full_model(
    const std::string& path,
    FactoredEmbLMHead& emb_lm,
    std::vector<SparseTransformerLayer>& tl,
    int dim, int layers, int heads, int vocab, float& density);
