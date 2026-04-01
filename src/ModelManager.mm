#import <Metal/Metal.h>
#include "ModelManager.h"
#include <fstream>
#include <iostream>
#include <vector>

void save_full_model(
    const std::string& path,
    const FactoredEmbLMHead& emb_lm,
    const std::vector<SparseTransformerLayer>& tl,
    int dim, int layers, int heads, int vocab, float density)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot save to " << path << "\n"; return; }

    uint32_t magic = 0x4A4C5832; // "JLX2"
    f.write((const char*)&magic, 4);
    f.write((const char*)&dim, sizeof(int));
    f.write((const char*)&layers, sizeof(int));
    f.write((const char*)&heads, sizeof(int));
    f.write((const char*)&vocab, sizeof(int));
    f.write((const char*)&density, sizeof(float));

    emb_lm.save(f);
    for (int i = 0; i < layers; ++i) {
        tl[i].attn->save(f);
        tl[i].ffn->save(f);
    }

    std::cout << "  [Save] Full model -> " << path << " ("
              << f.tellp() / (1024*1024) << " MB)\n";
}

bool load_full_model(
    const std::string& path,
    FactoredEmbLMHead& emb_lm,
    std::vector<SparseTransformerLayer>& tl,
    int dim, int layers, int heads, int vocab, float& density)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot load " << path << "\n"; return false; }

    uint32_t magic;
    f.read((char*)&magic, 4);
    if (magic != 0x4A4C5832) { std::cerr << "Bad magic (need JLX2)\n"; return false; }

    int d, l, h, v;
    f.read((char*)&d, sizeof(int));
    f.read((char*)&l, sizeof(int));
    f.read((char*)&h, sizeof(int));
    f.read((char*)&v, sizeof(int));
    f.read((char*)&density, sizeof(float));
    if (d != dim || l != layers || h != heads || v != vocab) {
        std::cerr << "Model mismatch\n";
        return false;
    }

    emb_lm.load(f);
    for (int i = 0; i < layers; ++i) {
        tl[i].attn->load(f);
        tl[i].ffn->load(f);
    }

    for (int i = 0; i < (int)tl.size(); ++i) {
        for (auto& w : tl[i].all_sparse_weights())
            w->sync_packed_weights();
    }

    std::cout << "  [Load] Full model <- " << path << "\n";
    return true;
}
