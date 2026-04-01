#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include "Tensor.h"
#include "Kernel.h"

// Memory usage check
size_t get_gpu_mb();
size_t get_rss_mb();

// BF16 helper
uint16_t fp32_to_bf16(float v);

// Positional encoding
std::vector<float> make_sinusoidal(int seq, int dim);

// In-place GPU addition
void gpu_add_inplace(Kernel& add_k, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

// Tensor clone (on unified memory)
std::shared_ptr<Tensor> tensor_clone(std::shared_ptr<Tensor> src);

// Learning rate scheduler
float cosine_lr(float base_lr, int step, int warmup, int total);
