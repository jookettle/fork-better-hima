#import <Metal/Metal.h>
#include "RMSNorm.h"
#include "MetalContext.h"
#include "Kernel.h"
#include <algorithm>

RMSNorm::RMSNorm() : eps(1e-6f) {}

void RMSNorm::init() {
    fwd_k = std::make_unique<Kernel>("rms_norm_forward",  "kernels/ops.metal");
    bwd_k = std::make_unique<Kernel>("rms_norm_backward", "kernels/ops.metal");
}

std::shared_ptr<Tensor> RMSNorm::forward(std::shared_ptr<Tensor> x) {
    int rows = (int)(x->size() / x->getShape().back());
    uint32_t d = (uint32_t)x->getShape().back();
    auto out = std::make_shared<Tensor>(x->getShape(), DType::Float32);
    auto enc = CommandBatch::get().encoder();
    [enc setComputePipelineState:fwd_k->getPipelineState()];
    [enc setBuffer:out->getBuffer() offset:0 atIndex:0];
    [enc setBuffer:x->getBuffer()   offset:0 atIndex:1];
    [enc setBytes:&d   length:4 atIndex:2];
    [enc setBytes:&eps length:4 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(rows, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(std::min(rows, 256), 1, 1)];
    return out;
}

std::shared_ptr<Tensor> RMSNorm::backward(std::shared_ptr<Tensor> dy,
                                 std::shared_ptr<Tensor> x_orig) {
    int rows = (int)(dy->size() / dy->getShape().back());
    uint32_t d = (uint32_t)dy->getShape().back();
    auto dx = std::make_shared<Tensor>(dy->getShape(), DType::Float32);
    auto enc = CommandBatch::get().encoder();
    [enc setComputePipelineState:bwd_k->getPipelineState()];
    [enc setBuffer:dx->getBuffer()     offset:0 atIndex:0];
    [enc setBuffer:dy->getBuffer()     offset:0 atIndex:1];
    [enc setBuffer:x_orig->getBuffer() offset:0 atIndex:2];
    [enc setBytes:&d   length:4 atIndex:3];
    [enc setBytes:&eps length:4 atIndex:4];
    [enc dispatchThreads:MTLSizeMake(rows, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(std::min(rows, 256), 1, 1)];
    return dx;
}
