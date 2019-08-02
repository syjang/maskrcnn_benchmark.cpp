#pragma once
#include <torch/torch.h>
#include <map>
#include <vector>
#include "bounding_box.h"
#include "box_coder.h"
#include "matcher.h"

#include "rpn/anchor_generator.h"
#include "rpn/fcos/loss.h"
#include "rpn/fcos/inference.h"
#include "scale.h"

namespace rcnn
{
namespace modeling
{

class FCOSHeadImpl : public torch::nn::Module
{
public:
    FCOSHeadImpl(int64_t in_channels);
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> forward(std::vector<torch::Tensor> &x);

private:
    torch::nn::Conv2d cls_logits = nullptr;
    torch::nn::Conv2d bbox_pred = nullptr;
    torch::nn::Conv2d centerness = nullptr;
    torch::nn::Sequential cls_tower = nullptr;
    torch::nn::Sequential bbox_tower = nullptr;

    std::vector<layers::Scale> scales;
};
TORCH_MODULE(FCOSHead);

class FCOSModuleImpl : public torch::nn::Module
{
public:
    FCOSModuleImpl(int64_t in_channels);
    //for train
    std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>>
    forward(rcnn::structures::ImageList &images, std::vector<torch::Tensor> &features, std::vector<rcnn::structures::BoxList> targets);

    //for inference and testing
    std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>>
    forward(rcnn::structures::ImageList &images, std::vector<torch::Tensor> &features);

private:
    std::vector<torch::Tensor>
    compute_locations(std::vector<torch::Tensor> &features);

    torch::Tensor
    compute_locations_per_level(int64_t h, int64_t w, int stride, torch::Device device);

    std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>>
    _forward_test(const std::vector<torch::Tensor> &locations,
                  const std::vector<torch::Tensor> &box_cls,
                  const std::vector<torch::Tensor> &box_regression,
                  const std::vector<torch::Tensor> &centerness,
                  const std::vector<std::pair<int64_t, int64_t>> &image_sizes);

    std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>>
    _forward_train(const std::vector<torch::Tensor> &locations,
                   const std::vector<torch::Tensor> &box_cls,
                   const std::vector<torch::Tensor> &box_regression,
                   const std::vector<torch::Tensor> &centerness,
                   const std::vector<rcnn::structures::BoxList> &targets);

private:
    FCOSHead head = nullptr;
    FCOSPostProcessor box_selector_test = nullptr;

    //todo make loss
    //loss_evaluator
    std::vector<int> fpn_strides;

    bool isTrain = false;
};
TORCH_MODULE(FCOSModule);

} // namespace modeling
} // namespace rcnn