// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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
class Scale : public torch::nn::Module
{
public:
    Scale(float init_value = 1.0);
    torch::Tensor forward(torch::Tensor input);

private:
    float scale = 1.0;
};
} // namespace layers
} // namespace rcnn