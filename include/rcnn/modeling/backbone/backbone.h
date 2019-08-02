#pragma once
#include <torch/torch.h>

namespace rcnn
{
namespace modeling
{

class BackboneImpl : public torch::nn::Module
{

public:
  explicit BackboneImpl(torch::nn::Sequential backbone, int64_t out_channels);
  std::vector<torch::Tensor> forward(torch::Tensor x);
  int64_t get_out_channels();
  std::shared_ptr<BackboneImpl> clone(torch::optional<torch::Device> device = torch::nullopt) const;

private:
  torch::nn::Sequential backbone_;
  int64_t out_channels_;
};

TORCH_MODULE(Backbone);

Backbone BuildResnetBackbone();
Backbone BuildResnetFPNBackbone();
Backbone BuildBackbone();
Backbone BuildVoVNetFPNBackbone();
Backbone BuildResnetFpn_p3p7_backbone();

} // namespace modeling
} // namespace rcnn