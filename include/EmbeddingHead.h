#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include "Tensor.h"
#include "Kernel.h"

class FactoredEmbLMHead {
public:
    void init(int vocab_, int dim_, int k_);
    void sync_to_gpu();
    
    std::shared_ptr<Tensor> emb_forward(
        std::shared_ptr<Tensor> tok, const std::vector<float>& pos_enc,
        int B, int S);
    
    std::shared_ptr<Tensor> lm_forward(std::shared_ptr<Tensor> hidden, int BS);
    std::shared_ptr<Tensor> lm_backward(std::shared_ptr<Tensor> dlogits, int BS);
    
    void emb_backward(std::shared_ptr<Tensor> cg, std::shared_ptr<Tensor> tok, int BS);
    void adam_step(float lr, float accum_scale);
    
    void save(std::ostream& f) const;
    void load(std::istream& f);

private:
    int vocab, dim, k;

    // Sub-embedding: [vocab x k]
    std::vector<float> sub_emb;
    std::vector<float> sub_emb_m, sub_emb_v;
    std::shared_ptr<Tensor> sub_emb_gpu;

    // Embedding projection: [k x dim]
    std::vector<float> emb_proj;
    std::vector<float> emb_proj_m, emb_proj_v;
    std::shared_ptr<Tensor> emb_proj_gpu;

    // LM head projection: [dim x k]
    std::vector<float> lm_proj;
    std::vector<float> lm_proj_m, lm_proj_v;
    std::shared_ptr<Tensor> lm_proj_gpu;

    // Gradients
    std::vector<float> sub_emb_g;
    std::vector<int> sub_emb_g_count;
    std::shared_ptr<Tensor> emb_proj_g_gpu;
    std::shared_ptr<Tensor> lm_proj_g_gpu;
    std::shared_ptr<Tensor> sub_emb_lm_g_gpu;

    // Saved state
    std::shared_ptr<Tensor> last_hidden, last_projected;
    std::unique_ptr<Kernel> add_kernel;
    int step_count = 0;
};
