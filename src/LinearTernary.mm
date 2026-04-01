#include "LinearTernary.h"
#include <iostream>

LinearTernary::LinearTernary(int in_features, int out_features) 
    : in_features(in_features), out_features(out_features) {
    
    std::vector<int> shape = {out_features, in_features};
    weights = std::make_shared<Tensor>(shape, DType::Ternary);
    weight_grads = std::make_shared<Tensor>(shape, DType::Float32);
    
    const std::string kpath = "kernels/ops.metal";
    matmulKernel    = std::make_unique<Kernel>("dense_ternary_matmul", kpath);
    bwdInputKernel  = std::make_unique<Kernel>("dense_ternary_backward_input", kpath);
    bwdWeightKernel = std::make_unique<Kernel>("dense_ternary_backward_weight", kpath);
    
    initialize_weights();
}

void LinearTernary::initialize_weights() {
    std::default_random_engine gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Xavier initialization or similar scale
    float scale = sqrtf(6.0f / (in_features + out_features));
    
    std::vector<int> shape = {out_features, in_features};
    size_t num_elements = (size_t)in_features * out_features;
    
    unsigned char* packed_data = (unsigned char*)weights->data();
    std::memset(packed_data, 0, weights->bytes());
    
    for (size_t i = 0; i < num_elements; ++i) {
        float val = dist(gen) * scale;
        int t_val = 0;
        if (val > 0.05f) t_val = 1;      // Simple threshold for ternary
        else if (val < -0.05f) t_val = -1;
        
        // Packing: 2 bits per element
        // 0 -> 00, 1 -> 01, -1 -> 10
        unsigned char bits = 0;
        if (t_val == 1) bits = 1;
        else if (t_val == -1) bits = 2;
        
        size_t byte_idx = i / 4;
        size_t bit_pos = (i % 4) * 2;
        packed_data[byte_idx] |= (bits << bit_pos);
    }
}

std::vector<std::shared_ptr<Tensor>> LinearTernary::forward(const std::vector<std::shared_ptr<Tensor>>& inputs) {
    if (inputs.empty()) return {};
    auto input = inputs[0];
    last_input = input;
    
    int batch_size = input->getShape()[0];
    std::vector<int> out_shape = {batch_size, out_features};
    auto output = std::make_shared<Tensor>(out_shape, DType::Float32);
    
    struct Params { uint32_t b, i, o; float scale; } p = {
        (uint32_t)batch_size, (uint32_t)in_features, (uint32_t)out_features, 1.0f };

    matmulKernel->dispatch({input.get(), weights.get()}, {output.get()}, &p, sizeof(p));
    
    return {output};
}

std::vector<std::shared_ptr<Tensor>> LinearTernary::backward(const std::vector<std::shared_ptr<Tensor>>& grad_outputs) {
    if (grad_outputs.empty() || !last_input) return {};
    auto grad_out = grad_outputs[0];
    
    int batch_size = grad_out->getShape()[0];
    std::vector<int> gi_shape = {batch_size, in_features};
    auto grad_in = std::make_shared<Tensor>(gi_shape, DType::Float32);
    
    struct Params { uint32_t b, i, o; float scale; } p = {
        (uint32_t)batch_size, (uint32_t)in_features, (uint32_t)out_features, 1.0f };

    // 1. Gradient for input
    bwdInputKernel->dispatch({grad_out.get(), weights.get()}, {grad_in.get()}, &p, sizeof(p));
    
    // 2. Gradient for weights
    if (!weight_grads) {
        weight_grads = std::make_shared<Tensor>(std::vector<int>{out_features, in_features}, DType::Float32);
        std::memset(weight_grads->data(), 0, weight_grads->bytes());
    }
    
    // We use a temporary buffer for current batch weight gradient then add it
    auto grad_w_batch = std::make_shared<Tensor>(weight_grads->getShape(), DType::Float32);
    bwdWeightKernel->dispatch({grad_out.get(), last_input.get()}, {grad_w_batch.get()}, &p, sizeof(p));
    
    // Accumulate on CPU (or we could add a kernel for this, but for now CPU is fine for sync)
    float* dst = (float*)weight_grads->data();
    float* src = (float*)grad_w_batch->data();
    for (size_t i = 0; i < weight_grads->size(); ++i) dst[i] += src[i];
    
    return {grad_in};
}

void LinearTernary::clear_activations() {
    last_input.reset();
}

void LinearTernary::save(std::ostream& os) const {
    os.write((const char*)&in_features,  sizeof(int));
    os.write((const char*)&out_features, sizeof(int));
    if (weights) os.write((const char*)weights->data(), weights->bytes());
}

void LinearTernary::load(std::istream& is) {
    is.read((char*)&in_features,  sizeof(int));
    is.read((char*)&out_features, sizeof(int));
    if (weights) is.read((char*)weights->data(), weights->bytes());
}