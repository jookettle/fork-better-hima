#import <Metal/Metal.h>
#include <mach/mach.h>
#include <cmath>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "Utils.h"
#include "MetalContext.h"
#include "Kernel.h"
#include "Tensor.h"

size_t get_gpu_mb() {
    id<MTLDevice> dev = MetalContext::getInstance().getDevice();
    return [dev currentAllocatedSize] / (1024 * 1024);
}

size_t get_rss_mb() {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS)
        return info.resident_size / (1024 * 1024);
    return 0;
}

uint16_t fp32_to_bf16(float v) {
    uint32_t b = *(uint32_t*)&v;
    b += 0x7FFFu + ((b >> 16) & 1u);
    return (uint16_t)(b >> 16);
}

std::vector<float> make_sinusoidal(int seq, int dim) {
    std::vector<float> pe((size_t)seq * dim, 0.f);
    for (int pos = 0; pos < seq; ++pos)
        for (int i = 0; i < dim; i += 2) {
            float f = 1.f / powf(10000.f, (float)i / dim);
            pe[(size_t)pos * dim + i] = sinf(pos * f);
            if (i + 1 < dim) pe[(size_t)pos * dim + i + 1] = cosf(pos * f);
        }
    return pe;
}

void gpu_add_inplace(Kernel& add_k, std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    uint32_t n = (uint32_t)a->size();
    auto enc = CommandBatch::get().encoder();
    [enc setComputePipelineState:add_k.getPipelineState()];
    [enc setBuffer:a->getBuffer() offset:0 atIndex:0];
    [enc setBuffer:b->getBuffer() offset:0 atIndex:1];
    [enc setBytes:&n length:4 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

std::shared_ptr<Tensor> tensor_clone(std::shared_ptr<Tensor> src) {
    auto dst = std::make_shared<Tensor>(src->getShape(), DType::Float32);
    // Explicitly copy Metal buffer contents to the new tensor's buffer
    std::memcpy(dst->data(), src->data(), src->bytes());
    return dst;
}

float cosine_lr(float base_lr, int step, int warmup, int total) {
    if (step < warmup) return base_lr * (float)step / (float)warmup;
    float progress = (float)(step - warmup) / (float)std::max(total - warmup, 1);
    return base_lr * 0.5f * (1.0f + cosf((float)M_PI * progress));
}
