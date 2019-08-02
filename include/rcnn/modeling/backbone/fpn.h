#pragma once
#include <torch/torch.h>
#include <vector>

namespace rcnn
{
namespace modeling
{

using ConvFunction = torch::nn::Sequential (*)(bool, int64_t, int64_t, int64_t, int64_t, int64_t);

class LastLevelP6P7Impl : public torch::nn::Module
{
public:
  LastLevelP6P7Impl(const int64_t &in_channels, const int64_t &out_channels);
  std::vector<torch::Tensor> forward(torch::Tensor c5, torch::Tensor p5);

private:
  torch::nn::Conv2d p6 = nullptr;
  torch::nn::Conv2d p7 = nullptr;

  bool useP5 = false;
};

TORCH_MODULE(LastLevelP6P7);

class FPNImpl : public torch::nn::Module
{

public:
  FPNImpl(const bool &use_relu, const std::vector<int64_t> &in_channels_list, const int64_t &out_channels, ConvFunction conv_block);
  std::vector<torch::Tensor> forward(std::vector<torch::Tensor> &x);
  std::shared_ptr<FPNImpl> clone(torch::optional<torch::Device> device = torch::nullopt) const;

  const bool use_relu_;
  const std::vector<int64_t> in_channels_list_;
  const int64_t out_channels_;
  ConvFunction conv_block_;

private:
  std::vector<torch::nn::Sequential> inner_blocks_;
  std::vector<torch::nn::Sequential> layer_blocks_;
};

TORCH_MODULE(FPN);

class FPNLastMaxPoolImpl : public torch::nn::Module
{

public:
  FPNLastMaxPoolImpl(const bool use_relu, const std::vector<int64_t> in_channels_list, const int64_t out_channels, ConvFunction conv_block);
  std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);
  std::shared_ptr<FPNLastMaxPoolImpl> clone(torch::optional<torch::Device> device = torch::nullopt) const;

private:
  FPN fpn_;
};

TORCH_MODULE(FPNLastMaxPool);

class FPNLastLevelP6P7Impl : public torch::nn::Module
{

public:
  FPNLastLevelP6P7Impl(const bool &use_relu, const std::vector<int64_t> &in_channels_list, const int64_t &in_channels, const int64_t &out_channels, ConvFunction conv_block);
  std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);
  std::shared_ptr<FPNLastLevelP6P7Impl> clone(torch::optional<torch::Device> device = torch::nullopt) const;

private:
  torch::nn::Conv2d p6 = nullptr;
  torch::nn::Conv2d p7 = nullptr;
  int64_t in_channels_;
  FPN fpn_ = nullptr;
  bool useP5 = false;
};

TORCH_MODULE(FPNLastLevelP6P7);

} // namespace modeling
} // namespace rcnn