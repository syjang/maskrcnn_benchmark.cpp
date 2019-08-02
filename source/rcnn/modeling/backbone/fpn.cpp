#include "backbone/fpn.h"
#include <algorithm>

namespace rcnn
{
namespace modeling
{

LastLevelP6P7Impl::LastLevelP6P7Impl(const int64_t &in_channels, const int64_t &out_channels)
{
  p6 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).stride(2).padding(1));
  p7 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).stride(2).padding(1));

  register_module("p6", p6);
  register_module("p7", p6);

  torch::nn::init::kaiming_uniform_(p6->weight, 1);
  torch::nn::init::kaiming_uniform_(p7->weight, 1);

  torch::nn::init::constant_(p6->bias, 0);
  torch::nn::init::constant_(p7->bias, 0);

  useP5 = in_channels == out_channels;
}

std::vector<torch::Tensor> LastLevelP6P7Impl::forward(torch::Tensor c5, torch::Tensor p5)
{
  auto x = useP5 ? p5 : c5;
  auto p_6 = p6->forward(x);
  auto p_7 = p7->forward(torch::relu(p_6));

  return {p_6, p_7};
}

FPNImpl::FPNImpl(const bool &use_relu, const std::vector<int64_t> &in_channels_list, const int64_t &out_channels, ConvFunction conv_block)
    : use_relu_(use_relu),
      in_channels_list_(in_channels_list),
      out_channels_(out_channels),
      conv_block_(conv_block)
{
  for (int i = 0; i < in_channels_list.size(); ++i)
  {
    inner_blocks_.push_back(register_module("fpn_inner" + std::to_string(i + 1), conv_block(use_relu, in_channels_list[i], out_channels, 1, 1, 1)));
    layer_blocks_.push_back(register_module("fpn_layer" + std::to_string(i + 1), conv_block(use_relu, out_channels, out_channels, 3, 1, 1)));
  }
};

std::vector<torch::Tensor> FPNImpl::forward(std::vector<torch::Tensor> &x)
{
  std::vector<torch::Tensor> results;
  torch::Tensor inner_top_down;
  torch::Tensor inner_lateral;
  torch::Tensor last_inner = inner_blocks_[3]->forward(x[3]);
  results.push_back(layer_blocks_[3]->forward(last_inner));
  for (int i = inner_blocks_.size() - 2; i >= 0; --i)
  {
    inner_top_down = torch::upsample_nearest2d(last_inner, {last_inner.size(2) * 2, last_inner.size(3) * 2});
    inner_lateral = inner_blocks_[i]->forward(x[i]);
    last_inner = inner_top_down + inner_lateral;
    results.push_back(layer_blocks_[i]->forward(last_inner));
  }
  std::reverse(results.begin(), results.end());
  return results;
}

std::shared_ptr<FPNImpl> FPNImpl::clone(torch::optional<torch::Device> device) const
{
  torch::NoGradGuard no_grad;
  std::shared_ptr<FPNImpl> copy = std::make_shared<FPNImpl>(use_relu_, in_channels_list_, out_channels_, conv_block_);
  for (auto &i : copy->named_parameters())
  {
    i.value().copy_(named_parameters()[i.key()]);
  }
  if (device.has_value())
    copy->to(device.value());
  return copy;
}

FPNLastMaxPoolImpl::FPNLastMaxPoolImpl(bool use_relu, const std::vector<int64_t> in_channels_list, const int64_t out_channels, ConvFunction conv_block)
    : fpn_(register_module("fpn", FPN(use_relu, in_channels_list, out_channels, conv_block))){};

std::vector<torch::Tensor> FPNLastMaxPoolImpl::forward(std::vector<torch::Tensor> x)
{
  std::vector<torch::Tensor> results = fpn_->forward(x);
  results.push_back(torch::max_pool2d(results.back(), 1, 2, 0));
  return results;
}

std::shared_ptr<FPNLastMaxPoolImpl> FPNLastMaxPoolImpl::clone(torch::optional<torch::Device> device) const
{
  torch::NoGradGuard no_grad;
  std::shared_ptr<FPNLastMaxPoolImpl> copy = std::make_shared<FPNLastMaxPoolImpl>(fpn_->use_relu_, fpn_->in_channels_list_, fpn_->out_channels_, fpn_->conv_block_);
  auto named_params = named_parameters();
  for (auto &i : copy->named_parameters())
  {
    i.value().copy_(named_params[i.key()]);
  }
  if (device.has_value())
    copy->to(device.value());
  return copy;
}

FPNLastLevelP6P7Impl::FPNLastLevelP6P7Impl(const bool &use_relu, const std::vector<int64_t> &in_channels_list, const int64_t &in_channels, const int64_t &out_channels, ConvFunction conv_block)
    : fpn_(register_module("fpn", FPN(use_relu, in_channels_list, out_channels, conv_block))), in_channels_(in_channels)
{
  p6 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).stride(2).padding(1));
  p7 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).stride(2).padding(1));

  register_module("p6", p6);
  register_module("p7", p6);

  torch::nn::init::kaiming_uniform_(p6->weight, 1);
  torch::nn::init::kaiming_uniform_(p7->weight, 1);

  torch::nn::init::constant_(p6->bias, 0);
  torch::nn::init::constant_(p7->bias, 0);

  useP5 = in_channels == out_channels;
}

std::vector<torch::Tensor> FPNLastLevelP6P7Impl::forward(std::vector<torch::Tensor> x)
{
  std::vector<torch::Tensor> results = fpn_->forward(x);
  auto c5 = x.back();
  auto p5 = results.back();

  auto x_t = useP5 ? p5 : c5;
  auto p_6 = p6->forward(x_t);
  auto p_7 = p7->forward(torch::relu(p_6));

  results.push_back(p_6);
  results.push_back(p_7);
  return results;
}
std::shared_ptr<FPNLastLevelP6P7Impl> FPNLastLevelP6P7Impl::clone(torch::optional<torch::Device> device) const
{
  torch::NoGradGuard no_grad;
  std::shared_ptr<FPNLastLevelP6P7Impl> copy = std::make_shared<FPNLastLevelP6P7Impl>(fpn_->use_relu_, fpn_->in_channels_list_, in_channels_, fpn_->out_channels_, fpn_->conv_block_);
  auto named_params = named_parameters();
  for (auto &i : copy->named_parameters())
  {
    i.value().copy_(named_params[i.key()]);
  }
  if (device.has_value())
    copy->to(device.value());
  return copy;
}

} // namespace modeling
} // namespace rcnn