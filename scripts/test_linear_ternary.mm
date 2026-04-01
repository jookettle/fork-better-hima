#include "LinearTernary.h"
#include "MetalContext.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

void test_forward_backward() {
    std::cout << "Starting LinearTernary Test..." << std::endl;
    
    int in_features = 4;
    int out_features = 2;
    int batch_size = 1;
    
    LinearTernary layer(in_features, out_features);
    
    // Create dummy input [batch, in_features]
    std::vector<int> in_shape = {batch_size, in_features};
    auto input = std::make_shared<Tensor>(in_shape, DType::Float32);
    float* in_ptr = (float*)input->data();
    for (int i = 0; i < in_features; ++i) in_ptr[i] = (float)(i + 1);
    
    // Forward
    std::cout << "Running Forward..." << std::endl;
    auto outputs = layer.forward({input});
    auto output = outputs[0];
    
    float* out_ptr = (float*)output->data();
    
    std::cout << "Output: ";
    for (int i = 0; i < out_features; ++i) std::cout << out_ptr[i] << " ";
    std::cout << std::endl;
    
    // Backward
    std::cout << "Running Backward..." << std::endl;
    std::vector<int> out_shape = {batch_size, out_features};
    auto grad_out = std::make_shared<Tensor>(out_shape, DType::Float32);
    float* go_ptr = (float*)grad_out->data();
    for (int i = 0; i < out_features; ++i) go_ptr[i] = 1.0f; // Unit gradient
    
    auto grad_ins = layer.backward({grad_out});
    auto grad_in = grad_ins[0];
    
    float* gi_ptr = (float*)grad_in->data();
    
    std::cout << "Grad Input: ";
    for (int i = 0; i < in_features; ++i) std::cout << gi_ptr[i] << " ";
    std::cout << std::endl;
    
    auto params = layer.param_gradients();
    auto grad_w = params[0];
    float* gw_ptr = (float*)grad_w->data();
    
    std::cout << "Grad Weights (first row): ";
    for (int i = 0; i < in_features; ++i) std::cout << gw_ptr[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Test Complete!" << std::endl;
}

int main() {
    @autoreleasepool {
        test_forward_backward();
    }
    return 0;
}
