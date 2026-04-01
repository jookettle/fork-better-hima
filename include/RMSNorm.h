#pragma once
#include <memory>
#include "Tensor.h"
#include "Kernel.h"

class RMSNorm {
public:
    RMSNorm();
    void init();
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> dy, std::shared_ptr<Tensor> x_orig);

private:
    float eps;
    std::unique_ptr<Kernel> fwd_k, bwd_k;
};
