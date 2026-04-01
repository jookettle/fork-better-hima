#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "EmbeddingHead.h"
#include "MetalContext.h"
#include "Tensor.h"
#include "Kernel.h"
#include <cmath>
#include <algorithm>
#include <cstring>

static void mps_matmul(id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                       int M, int N, int K, bool transA = false, bool transB = false) {
    auto& ctx = MetalContext::getInstance();
    id<MTLCommandBuffer> cmd = [ctx.getCommandQueue() commandBuffer];
    int ldA = transA ? M : K;
    int ldB = transB ? K : N;
    MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:(transA?K:M) columns:(transA?M:K) rowBytes:ldA*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:(transB?N:K) columns:(transB?K:N) rowBytes:ldB*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N*4 dataType:MPSDataTypeFloat32];
    MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:A descriptor:dA];
    MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:B descriptor:dB];
    MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:C descriptor:dC];
    MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
        initWithDevice:ctx.getDevice() transposeLeft:transA transposeRight:transB
        resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
    [gemm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
    [cmd commit]; [cmd waitUntilCompleted];
}

void FactoredEmbLMHead::init(int vocab_, int dim_, int k_) {
    vocab = vocab_; dim = dim_; k = k_;

    sub_emb.resize((size_t)vocab * k);
    sub_emb_m.assign((size_t)vocab * k, 0.f);
    sub_emb_v.assign((size_t)vocab * k, 0.f);
    sub_emb_g.assign((size_t)vocab * k, 0.f);
    sub_emb_g_count.assign(vocab, 0);
    sub_emb_gpu = std::make_shared<Tensor>(std::vector<int>{vocab, k}, DType::Float32);

    emb_proj.resize((size_t)k * dim);
    emb_proj_m.assign((size_t)k * dim, 0.f);
    emb_proj_v.assign((size_t)k * dim, 0.f);
    emb_proj_gpu = std::make_shared<Tensor>(std::vector<int>{k, dim}, DType::Float32);
    emb_proj_g_gpu = std::make_shared<Tensor>(std::vector<int>{k, dim}, DType::Float32);

    lm_proj.resize((size_t)dim * k);
    lm_proj_m.assign((size_t)dim * k, 0.f);
    lm_proj_v.assign((size_t)dim * k, 0.f);
    lm_proj_gpu = std::make_shared<Tensor>(std::vector<int>{dim, k}, DType::Float32);
    lm_proj_g_gpu = std::make_shared<Tensor>(std::vector<int>{dim, k}, DType::Float32);

    sub_emb_lm_g_gpu = std::make_shared<Tensor>(std::vector<int>{k, vocab}, DType::Float32);

    auto xinit = [](std::vector<float>& w, int fan_in, int fan_out) {
        float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
        uint32_t seed = 42 + fan_in * 31 + fan_out * 17;
        for (auto& v : w) {
            seed = seed * 1664525u + 1013904223u;
            v = ((float)(seed >> 1) / (float)0x7FFFFFFF * 2.0f - 1.0f) * limit;
        }
    };
    xinit(sub_emb, vocab, k);
    xinit(emb_proj, k, dim);
    xinit(lm_proj, dim, k);

    sync_to_gpu();
    add_kernel = std::make_unique<Kernel>("elementwise_add", "kernels/ops.metal");
    std::memset(emb_proj_g_gpu->data(), 0, emb_proj_g_gpu->bytes());
    std::memset(lm_proj_g_gpu->data(), 0, lm_proj_g_gpu->bytes());
}

void FactoredEmbLMHead::sync_to_gpu() {
    std::memcpy(sub_emb_gpu->data(), sub_emb.data(), sub_emb.size() * 4);
    std::memcpy(emb_proj_gpu->data(), emb_proj.data(), emb_proj.size() * 4);
    std::memcpy(lm_proj_gpu->data(), lm_proj.data(), lm_proj.size() * 4);
}

std::shared_ptr<Tensor> FactoredEmbLMHead::emb_forward(
    std::shared_ptr<Tensor> tok, const std::vector<float>& pos_enc, int B, int S) {
    int BS = B * S;
    auto sub_out = std::make_shared<Tensor>(std::vector<int>{BS, k}, DType::Float32);
    float* sp = (float*)sub_out->data();
    int32_t* tp = (int32_t*)tok->data();
    for (int i = 0; i < BS; ++i) {
        int v = std::max(0, std::min(tp[i], vocab - 1));
        std::memcpy(sp + i * k, sub_emb.data() + (size_t)v * k, k * sizeof(float));
    }
    auto out = std::make_shared<Tensor>(std::vector<int>{BS, dim}, DType::Float32);
    mps_matmul(sub_out->getBuffer(), emb_proj_gpu->getBuffer(), out->getBuffer(), BS, dim, k);
    float* op = (float*)out->data();
    for (int i = 0; i < BS; ++i) {
        const float* pe = pos_enc.data() + (size_t)(i % S) * dim;
        for (int d = 0; d < dim; ++d) op[i * dim + d] += pe[d];
    }
    return out;
}

std::shared_ptr<Tensor> FactoredEmbLMHead::lm_forward(std::shared_ptr<Tensor> hidden, int BS) {
    last_hidden = hidden;
    CommandBatch::get().commit_and_wait();
    last_projected = std::make_shared<Tensor>(std::vector<int>{BS, k}, DType::Float32);
    mps_matmul(hidden->getBuffer(), lm_proj_gpu->getBuffer(), last_projected->getBuffer(), BS, k, dim);
    auto logits = std::make_shared<Tensor>(std::vector<int>{BS, vocab}, DType::Float32);
    mps_matmul(last_projected->getBuffer(), sub_emb_gpu->getBuffer(), logits->getBuffer(), BS, vocab, k, false, true);
    CommandBatch::get().begin();
    return logits;
}

std::shared_ptr<Tensor> FactoredEmbLMHead::lm_backward(std::shared_ptr<Tensor> dlogits, int BS) {
    CommandBatch::get().commit_and_wait();
    auto d_proj = std::make_shared<Tensor>(std::vector<int>{BS, k}, DType::Float32);
    mps_matmul(dlogits->getBuffer(), sub_emb_gpu->getBuffer(), d_proj->getBuffer(), BS, k, vocab);
    auto dx = std::make_shared<Tensor>(std::vector<int>{BS, dim}, DType::Float32);
    mps_matmul(d_proj->getBuffer(), lm_proj_gpu->getBuffer(), dx->getBuffer(), BS, dim, k, false, true);
    auto g_lm = std::make_shared<Tensor>(std::vector<int>{dim, k}, DType::Float32);
    mps_matmul(last_hidden->getBuffer(), d_proj->getBuffer(), g_lm->getBuffer(), dim, k, BS, true, false);
    auto g_sub = std::make_shared<Tensor>(std::vector<int>{k, vocab}, DType::Float32);
    mps_matmul(last_projected->getBuffer(), dlogits->getBuffer(), g_sub->getBuffer(), k, vocab, BS, true, false);

    CommandBatch::get().begin();
    {
        uint32_t n = dim * k;
        auto enc = CommandBatch::get().encoder();
        [enc setComputePipelineState:add_kernel->getPipelineState()];
        [enc setBuffer:lm_proj_g_gpu->getBuffer() offset:0 atIndex:0];
        [enc setBuffer:g_lm->getBuffer()          offset:0 atIndex:1];
        [enc setBytes:&n length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
    {
        auto enc = CommandBatch::get().encoder();
        uint32_t n = k * vocab;
        [enc setComputePipelineState:add_kernel->getPipelineState()];
        [enc setBuffer:sub_emb_lm_g_gpu->getBuffer() offset:0 atIndex:0];
        [enc setBuffer:g_sub->getBuffer()              offset:0 atIndex:1];
        [enc setBytes:&n length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
    last_hidden.reset(); last_projected.reset();
    return dx;
}

void FactoredEmbLMHead::emb_backward(std::shared_ptr<Tensor> cg, std::shared_ptr<Tensor> tok, int BS) {
    CommandBatch::get().commit_and_wait();
    auto g_k = std::make_shared<Tensor>(std::vector<int>{BS, k}, DType::Float32);
    mps_matmul(cg->getBuffer(), emb_proj_gpu->getBuffer(), g_k->getBuffer(), BS, k, dim, false, true);
    auto sub_out = std::make_shared<Tensor>(std::vector<int>{BS, k}, DType::Float32);
    float* sp = (float*)sub_out->data();
    int32_t* tp = (int32_t*)tok->data();
    for (int i = 0; i < BS; ++i) {
        int v = std::max(0, std::min(tp[i], vocab - 1));
        std::memcpy(sp + i * k, sub_emb.data() + (size_t)v * k, k * sizeof(float));
    }
    auto g_ep = std::make_shared<Tensor>(std::vector<int>{k, dim}, DType::Float32);
    mps_matmul(sub_out->getBuffer(), cg->getBuffer(), g_ep->getBuffer(), k, dim, BS, true, false);
    CommandBatch::get().begin();
    {
        uint32_t n = k * dim;
        auto enc = CommandBatch::get().encoder();
        [enc setComputePipelineState:add_kernel->getPipelineState()];
        [enc setBuffer:emb_proj_g_gpu->getBuffer() offset:0 atIndex:0];
        [enc setBuffer:g_ep->getBuffer()           offset:0 atIndex:1];
        [enc setBytes:&n length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
    CommandBatch::get().commit_and_wait();
    float* gk = (float*)g_k->data();
    for (int i = 0; i < BS; ++i) {
        int v = std::max(0, std::min(tp[i], vocab - 1));
        float* dst = sub_emb_g.data() + (size_t)v * k;
        float* src = gk + (size_t)i * k;
        for (int d = 0; d < k; ++d) dst[d] += src[d];
        sub_emb_g_count[v]++;
    }
    CommandBatch::get().begin();
}

void FactoredEmbLMHead::adam_step(float lr, float accum_scale) {
    step_count++;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - powf(b1, (float)step_count);
    float bc2 = 1.0f - powf(b2, (float)step_count);
    CommandBatch::get().commit_and_wait();
    float* lm_g = (float*)sub_emb_lm_g_gpu->data();
    for (int v = 0; v < vocab; ++v) {
        float* e  = sub_emb.data()   + (size_t)v * k;
        float* mp = sub_emb_m.data() + (size_t)v * k;
        float* vp = sub_emb_v.data() + (size_t)v * k;
        float* ga = sub_emb_g.data() + (size_t)v * k;
        bool has_emb_g = (sub_emb_g_count[v] > 0);
        bool has_lm_g = false;
        for (int d = 0; d < k && !has_lm_g; ++d) if (lm_g[d * vocab + v] != 0.f) has_lm_g = true;
        if (!has_emb_g && !has_lm_g) continue;
        for (int d = 0; d < k; ++d) {
            float g = ga[d] * accum_scale + lm_g[d * vocab + v] * accum_scale;
            mp[d] = b1 * mp[d] + (1.f - b1) * g;
            vp[d] = b2 * vp[d] + (1.f - b2) * g * g;
            e[d] -= lr * (mp[d] / bc1) / (sqrtf(vp[d] / bc2) + eps);
            ga[d] = 0.f;
        }
        sub_emb_g_count[v] = 0;
    }
    std::memset(sub_emb_lm_g_gpu->data(), 0, sub_emb_lm_g_gpu->bytes());
    {
        float* ga = (float*)emb_proj_g_gpu->data();
        for (int i = 0; i < k * dim; ++i) {
            float g = ga[i] * accum_scale;
            emb_proj_m[i] = b1 * emb_proj_m[i] + (1.f - b1) * g;
            emb_proj_v[i] = b2 * emb_proj_v[i] + (1.f - b2) * g * g;
            emb_proj[i] -= lr * (emb_proj_m[i] / bc1) / (sqrtf(emb_proj_v[i] / bc2) + eps);
            ga[i] = 0.f;
        }
    }
    {
        float* ga = (float*)lm_proj_g_gpu->data();
        for (int i = 0; i < dim * k; ++i) {
            float g = ga[i] * accum_scale;
            lm_proj_m[i] = b1 * lm_proj_m[i] + (1.f - b1) * g;
            lm_proj_v[i] = b2 * lm_proj_v[i] + (1.f - b2) * g * g;
            lm_proj[i] -= lr * (lm_proj_m[i] / bc1) / (sqrtf(lm_proj_v[i] / bc2) + eps);
            ga[i] = 0.f;
        }
    }
    sync_to_gpu();
    CommandBatch::get().begin();
}

void FactoredEmbLMHead::save(std::ostream& f) const {
    f.write((const char*)&k, sizeof(int));
    f.write((const char*)sub_emb.data(), sub_emb.size() * 4);
    f.write((const char*)emb_proj.data(), emb_proj.size() * 4);
    f.write((const char*)lm_proj.data(), lm_proj.size() * 4);
}

void FactoredEmbLMHead::load(std::istream& f) {
    int k_file;
    f.read((char*)&k_file, sizeof(int));
    if (k_file != k) { std::cerr << "k mismatch: " << k_file << " vs " << k << "\n"; return; }
    f.read((char*)sub_emb.data(), sub_emb.size() * 4);
    f.read((char*)emb_proj.data(), emb_proj.size() * 4);
    f.read((char*)lm_proj.data(), lm_proj.size() * 4);
    sync_to_gpu();
}
