#import <Metal/Metal.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <random>

#include "MetalContext.h"
#include "Kernel.h"
#include "Tensor.h"
#include "DataLoader.h"
#include "SparseAttention.h"
#include "SparseFFN.h"
#include "SparseTernaryLinear.h"
#include "Adafactor.h"
#include "TierManager.h"
#include "runtime/Config.h"

// 메인 파일(main.mm)의 복잡도를 낮추고 각 컴포넌트를 독립적인 모듈로 분리하여 관리하기 위해 리팩토링
// 유틸리티, 레이어, 임베딩 및 모델 관리 로직을 별도의 .h/.mm 파일로 모듈화함.
#include "Utils.h"
#include "RMSNorm.h"
#include "EmbeddingHead.h"
#include "TransformerLayer.h"
#include "ModelManager.h"

int main() {
    Config& cfg = Config::getInstance();
    cfg.load("config.txt");
    cfg.print();

    const int dim     = cfg.getInt("DIM", 768);
    const int heads   = cfg.getInt("NUM_HEADS", 12);
    const int layers  = cfg.getInt("NUM_LAYERS", 6);
    const int seq     = cfg.getInt("SEQ_LEN", 128);
    const int batch   = cfg.getInt("BATCH_SIZE", 2);
    const float lr    = cfg.getFloat("LEARNING_RATE", 0.0003f);
    const int maxs    = cfg.getInt("MAX_STEPS", 100000);
    const int logi    = cfg.getInt("LOG_INTERVAL", 10);
    const int accum   = cfg.getInt("GRAD_ACCUM", 8);
    const bool overfit_test = cfg.getBool("OVERFIT_TEST", false);
    const std::string ddir = cfg.getString("DATASET_DIR", "data");
    const int block_size   = cfg.getInt("BLOCK_SIZE", 32);

    const int vocab    = 50257;
    const int BS       = batch * seq;
    const int ffn_dim  = dim * 4;
    const int warmup   = 50;
    const float accum_scale = 1.0f / (float)accum;

    const float density = cfg.getFloat("DENSITY", 0.05f);

    std::cout << "\n══════════════════════════════════════════════\n"
              << "  Sparse Ternary Transformer (Modular)\n"
              << "  dim=" << dim << " heads=" << heads
              << " ffn=" << ffn_dim << "\n"
              << "  seq=" << seq << " batch=" << batch << " BS=" << BS << "\n"
              << "  grad_accum=" << accum
              << " effective_BS=" << BS * accum << "\n"
              << "  density=" << density
              << " (NNZ/matrix ≈ " << (int)(dim * dim * density) << ")\n"
              << "  lr=" << lr << " warmup=" << warmup << " cosine_decay\n"
              << (overfit_test ? "  *** OVERFIT TEST ***\n" : "")
              << "══════════════════════════════════════════════\n\n";

    auto loader = std::make_unique<DataLoader>(ddir, "shard_*.bin", batch, seq);

    std::shared_ptr<Tensor> cached_ti, cached_tt;
    if (overfit_test) {
        auto [ti0, tt0] = loader->get_batch();
        cached_ti = std::make_shared<Tensor>(ti0->getShape(), DType::Int32);
        cached_tt = std::make_shared<Tensor>(tt0->getShape(), DType::Int32);
        std::memcpy(cached_ti->data(), ti0->data(), ti0->bytes());
        std::memcpy(cached_tt->data(), tt0->data(), tt0->bytes());
    }

    const int emb_k = cfg.getInt("EMB_K", 256);
    FactoredEmbLMHead emb_lm;
    emb_lm.init(vocab, dim, emb_k);
    auto pos_enc = make_sinusoidal(seq, dim);
    std::cout << "  Factored Emb+LM: vocab=" << vocab << " k=" << emb_k
              << " dim=" << dim << " (params: " << (vocab*emb_k + emb_k*dim + dim*emb_k)/1e6
              << "M vs " << 2.0*vocab*dim/1e6 << "M unfactored)\n";

    std::vector<SparseTransformerLayer> tl(layers);
    for (int i = 0; i < layers; ++i) {
        std::cout << "  Layer " << i << ": Sparse Ternary (density=" << density << ")\n";
        tl[i].init(dim, heads, block_size, density, ffn_dim);
    }

    RMSNorm final_norm; final_norm.init();

    AdafactorParams opt_params;
    opt_params.lr = lr;
    opt_params.beta1 = 0.9f;
    opt_params.decay_rate = 0.999f;
    opt_params.epsilon2 = 1e-8f;
    opt_params.warmup_steps = warmup;
    Adafactor optimizer(opt_params);

    int weight_idx = 0;
    std::vector<std::shared_ptr<SparseTernaryLinear>> all_sparse_layers;
    for (int i = 0; i < layers; ++i) {
        tl[i].hima_base_idx = weight_idx;
        auto weights = tl[i].all_sparse_weights();
        for (auto& w : weights) {
            optimizer.register_weight(w->get_master_weights_pos().get(), weight_idx++);
            all_sparse_layers.push_back(w);
        }
    }
    std::cout << "  Registered " << all_sparse_layers.size()
              << " sparse weight matrices\n";

    HiMAConfig hima_cfg;
    hima_cfg.hot_ratio  = 0.40f;
    hima_cfg.warm_ratio = 0.35f;
    hima_cfg.warm_update_interval = 10;
    hima_cfg.rebalance_interval   = 50;
    hima_cfg.grad_ema_alpha       = 0.1f;
    hima_cfg.min_steps_in_tier    = 20;
    hima_cfg.warm_lr_scale        = 0.1f;
    TierManager hima((int)all_sparse_layers.size(), hima_cfg);
    std::cout << "\n";

    Kernel ce_k("cross_entropy_loss", "kernels/ops.metal");
    Kernel add_k("elementwise_add", "kernels/ops.metal");

    std::cout << std::left
              << std::setw(7) << "Step" << std::setw(12) << "Loss"
              << std::setw(10) << "PPL" << std::setw(10) << "ms"
              << std::setw(10) << "Tok/s" << std::setw(6) << "GPU"
              << std::setw(6) << "RSS"
              << "Tokens\n" << std::string(70, '-') << "\n";
    std::cout.flush();

    double sms = -1;
    size_t total_tokens = 0;
    float best_loss = 999.f;
    int opt_step = 0;

    for (int step = 1; step <= maxs; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();
        float loss_sum = 0;

        opt_step++;
        float cur_lr = cosine_lr(lr, opt_step, warmup, maxs);
        optimizer.set_lr(cur_lr);

        for (int micro = 0; micro < accum; ++micro) { @autoreleasepool {
            std::shared_ptr<Tensor> ti, tt;
            if (overfit_test) {
                ti = std::make_shared<Tensor>(cached_ti->getShape(), DType::Int32);
                tt = std::make_shared<Tensor>(cached_tt->getShape(), DType::Int32);
                std::memcpy(ti->data(), cached_ti->data(), cached_ti->bytes());
                std::memcpy(tt->data(), cached_tt->data(), cached_tt->bytes());
            } else {
                auto batch_pair = loader->get_batch();
                ti = batch_pair.first;
                tt = batch_pair.second;
            }
            ti->reshape({BS}); tt->reshape({BS});

            // ── FORWARD ──
            auto r = emb_lm.emb_forward(ti, pos_enc, batch, seq);

            for (int li = 0; li < layers; ++li) {
                auto& L = tl[li];
                L.r_pre_attn = tensor_clone(r);

                CommandBatch::get().begin();
                auto xn = L.norm1.forward(r);
                xn->reshape({batch, seq, dim});
                auto attn_out = L.attn->forward({xn})[0];
                attn_out->reshape({BS, dim});
                gpu_add_inplace(add_k, r, attn_out);
                CommandBatch::get().commit_and_wait();

                L.r_pre_ffn = tensor_clone(r);

                CommandBatch::get().begin();
                auto xn2 = L.norm2.forward(r);
                xn2->reshape({batch, seq, dim});
                auto ffn_out = L.ffn->forward({xn2})[0];
                ffn_out->reshape({BS, dim});
                gpu_add_inplace(add_k, r, ffn_out);
                CommandBatch::get().commit_and_wait();
            }

            auto r_pre_final = tensor_clone(r);
            CommandBatch::get().begin();
            auto xf = final_norm.forward(r);
            auto logits = emb_lm.lm_forward(xf, BS);

            auto lbuf = std::make_shared<Tensor>(std::vector<int>{BS}, DType::Float32);
            auto glog = std::make_shared<Tensor>(std::vector<int>{BS, vocab}, DType::Float32);
            CommandBatch::get().begin();
            { auto enc = CommandBatch::get().encoder();
              [enc setComputePipelineState:ce_k.getPipelineState()];
              [enc setBuffer:lbuf->getBuffer()   offset:0 atIndex:0];
              [enc setBuffer:glog->getBuffer()   offset:0 atIndex:1];
              [enc setBuffer:logits->getBuffer() offset:0 atIndex:2];
              [enc setBuffer:tt->getBuffer()     offset:0 atIndex:3];
              uint32_t b32 = BS, v32 = vocab;
              [enc setBytes:&b32 length:4 atIndex:4];
              [enc setBytes:&v32 length:4 atIndex:5];
              [enc dispatchThreadgroups:MTLSizeMake(BS, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)]; }
            CommandBatch::get().commit_and_wait();

            { float* lp = (float*)lbuf->data();
              for (int i = 0; i < BS; ++i) loss_sum += std::isfinite(lp[i]) ? lp[i] : 20.f; }

            // ── BACKWARD ──
            auto cg = emb_lm.lm_backward(glog, BS);
            CommandBatch::get().begin();
            cg = final_norm.backward(cg, r_pre_final);
            CommandBatch::get().commit_and_wait();

            for (int li = layers - 1; li >= 0; --li) {
                auto& L = tl[li];
                Tier layer_tier = L.dominant_tier(hima);
                bool compute_weight_grad = (layer_tier != Tier::COLD);

                CommandBatch::get().begin();
                cg->reshape({batch, seq, dim});
                auto d_ffn = L.ffn->backward({cg})[0];
                d_ffn->reshape({BS, dim});
                auto d_norm2 = L.norm2.backward(d_ffn, L.r_pre_ffn);
                cg->reshape({BS, dim});
                gpu_add_inplace(add_k, cg, d_norm2);
                cg->reshape({batch, seq, dim});
                auto d_attn = L.attn->backward({cg})[0];
                d_attn->reshape({BS, dim});
                auto d_norm1 = L.norm1.backward(d_attn, L.r_pre_attn);
                cg->reshape({BS, dim});
                gpu_add_inplace(add_k, cg, d_norm1);
                CommandBatch::get().commit_and_wait();

                if (!compute_weight_grad) {
                    auto cold_weights = L.all_sparse_weights();
                    for (auto& w : cold_weights) w->clear_gradients();
                }
                L.clear_saved();
            }
            emb_lm.emb_backward(cg, ti, BS);
        }} // end micro-batch

        hima.step();

        for (int bi = 0; bi < (int)all_sparse_layers.size(); ++bi) {
            auto& w = all_sparse_layers[bi];
            float gnorm = 0.0f;
            int count = 0;
            auto gp = w->pos_gradients();
            auto gn = w->neg_gradients();
            if (gp) {
                float* gd = (float*)gp->data();
                int stride = std::max(1u, w->nnz_pos() / 256u);
                for (uint32_t k = 0; k < w->nnz_pos(); k += stride) { gnorm += gd[k] * gd[k]; count++; }
            }
            if (gn) {
                float* gd = (float*)gn->data();
                int stride = std::max(1u, w->nnz_neg() / 256u);
                for (uint32_t k = 0; k < w->nnz_neg(); k += stride) { gnorm += gd[k] * gd[k]; count++; }
            }
            if (count > 0) gnorm = sqrtf(gnorm / (float)count);
            hima.record_gradient(bi, gnorm);
        }

        float grad_norm_sq = 0.0f;
        for (int bi = 0; bi < (int)all_sparse_layers.size(); ++bi) {
            if (!hima.should_update(bi, opt_step)) continue;
            auto& w = all_sparse_layers[bi];
            auto gp = w->pos_gradients();
            auto gn = w->neg_gradients();
            if (gp) {
                float* gd = (float*)gp->data();
                for (uint32_t k = 0; k < w->nnz_pos(); ++k) grad_norm_sq += gd[k] * gd[k];
            }
            if (gn) {
                float* gd = (float*)gn->data();
                for (uint32_t k = 0; k < w->nnz_neg(); ++k) grad_norm_sq += gd[k] * gd[k];
            }
        }
        float grad_norm = sqrtf(grad_norm_sq);
        float clip_scale = (grad_norm > 1.0f && grad_norm > 0.f) ? 1.0f / grad_norm : 1.0f;

        CommandBatch::get().begin();
        for (int bi = 0; bi < (int)all_sparse_layers.size(); ++bi) {
            if (!hima.should_update(bi, opt_step)) continue;
            float lr_mul = hima.lr_scale(bi);
            float orig_lr = optimizer.get_lr();
            optimizer.set_lr(orig_lr * lr_mul);
            all_sparse_layers[bi]->fused_adam_update(optimizer, accum_scale, clip_scale, opt_step);
            optimizer.set_lr(orig_lr);
        }
        CommandBatch::get().commit_and_wait();

        for (int bi = 0; bi < (int)all_sparse_layers.size(); ++bi) {
            if (hima.should_update(bi, opt_step)) all_sparse_layers[bi]->sync_packed_weights();
        }
        for (auto& w : all_sparse_layers) w->clear_gradients();

        if (opt_step > warmup && opt_step % hima_cfg.rebalance_interval == 0) {
            int moves = hima.rebalance(opt_step);
            if (moves > 0 || opt_step % 200 == 0) hima.print_status();
        }

        emb_lm.adam_step(cur_lr, accum_scale);

        if (opt_step > 0 && opt_step % 500 == 0) {
            for (auto& w : all_sparse_layers) w->resparsify();
            std::cout << "  [Resparsify] step=" << opt_step << "\n";
        }

        float loss = loss_sum / (float)(BS * accum);
        if (!std::isfinite(loss)) continue;
        if (loss < best_loss) best_loss = loss;
        total_tokens += (size_t)BS * accum;

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (sms < 0) sms = ms; else sms = 0.95 * sms + 0.05 * ms;

        if (step % logi == 0 || step <= 5) {
            double ppl = exp(std::min((double)loss, 20.0));
            double toks = (double)(BS * accum) / (sms * 1e-3);
            std::cout << std::left << std::setw(7) << step
                      << std::fixed << std::setprecision(4) << std::setw(12) << loss
                      << std::setprecision(1) << std::setw(10) << ppl
                      << std::setprecision(0) << std::setw(10) << ms
                      << std::setprecision(0) << std::setw(10) << toks
                      << std::setw(6) << get_gpu_mb() << std::setw(6) << get_rss_mb()
                      << total_tokens << "\n";
            std::cout.flush();
        }
        if (step % 200 == 0) {
            std::cout << "  >>> step=" << step << " loss=" << std::fixed << std::setprecision(4) << loss 
                      << " best=" << best_loss << " lr=" << std::setprecision(6) << cur_lr << "\n";
            hima.print_status();
        }

        if (step % 1000 == 0) {
            std::string ckpt_dir = cfg.getString("CHECKPOINT_DIR", "checkpoints/");
            system(("mkdir -p " + ckpt_dir).c_str());
            save_full_model(ckpt_dir + "model_step_" + std::to_string(step) + ".jlx",
                            emb_lm, tl, dim, layers, heads, vocab, density);
            optimizer.save_state(ckpt_dir + "opt_step_" + std::to_string(step) + ".bin");
        }
    }

    std::string ckpt_dir = cfg.getString("CHECKPOINT_DIR", "checkpoints/");
    system(("mkdir -p " + ckpt_dir).c_str());
    save_full_model(ckpt_dir + "model_final.jlx", emb_lm, tl, dim, layers, heads, vocab, density);

    std::cout << "\n  Training done. best_loss=" << std::fixed << std::setprecision(4) << best_loss 
              << " tokens=" << total_tokens << "\n  Final memory: GPU=" << get_gpu_mb()
              << "MB RSS=" << get_rss_mb() << "MB\n";
    return 0;
}