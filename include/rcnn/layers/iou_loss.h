#pragma once
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif

namespace rcnn
{
namespace layers
{
class IOULoss : public torch::nn::Module
{
public:
    torch::Tensor forward(torch::Tensor pred, torch::Tensor target, torch::Tensor weight);
};
} // namespace layers
} // namespace rcnn