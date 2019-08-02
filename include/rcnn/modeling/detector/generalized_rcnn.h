#pragma once
#include <torch/torch.h>
#include <cassert>

#include <image_list.h>
#include <bounding_box.h>
#include <defaults.h>

#include "backbone/backbone.h"
#include "rpn/rpn.h"
#include "roi_heads/roi_heads.h"
#include "rpn/fcos/fcos.h"

namespace rcnn
{
namespace modeling
{

class GeneralizedRCNNImpl : public torch::nn::Module
{

public:
  GeneralizedRCNNImpl();
  std::shared_ptr<GeneralizedRCNNImpl> clone(torch::optional<torch::Device> device = torch::nullopt) const;

  template <typename T>
  T forward(std::vector<torch::Tensor> images, std::vector<rcnn::structures::BoxList> targets);

  template <typename T>
  T forward(rcnn::structures::ImageList images, std::vector<rcnn::structures::BoxList> targets);

  std::vector<rcnn::structures::BoxList> forward(std::vector<torch::Tensor> images);
  std::vector<rcnn::structures::BoxList> forward(rcnn::structures::ImageList images);

private:
  Backbone backbone;
  RPNModule rpn;
  FCOSModule fcos;
  CombinedROIHeads roi_heads;
  bool use_fcos = false;
};

TORCH_MODULE(GeneralizedRCNN);

template <>
std::map<std::string, torch::Tensor> GeneralizedRCNNImpl::forward(std::vector<torch::Tensor> images, std::vector<rcnn::structures::BoxList> targets);

template <>
std::map<std::string, torch::Tensor> GeneralizedRCNNImpl::forward(rcnn::structures::ImageList images, std::vector<rcnn::structures::BoxList> targets);

template <>
std::vector<rcnn::structures::BoxList> GeneralizedRCNNImpl::forward(rcnn::structures::ImageList images, std::vector<rcnn::structures::BoxList> targets);

} // namespace modeling
} // namespace rcnn