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

namespace rcnn
{
namespace modeling
{

class FCOSHead : public torch::nn::Module
{
public:
    FCOSHead(int64_t in_channels);
    std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> forward(std::vector<torch::Tensor> &x);

private:
};

class FCOSModule : public torch::nn::Module
{
public:
    FCOSModule(int64_t in_channels);
    std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> forward(rcnn::structures::ImageList &images, std::vector<torch::Tensor> &features, std::vector<rcnn::structures::BoxList> targets);
    std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>> forward(rcnn::structures::ImageList &images, std::vector<torch::Tensor> &features);

private:
    FCOSHead head;
};

} // namespace modeling
} // namespace rcnn