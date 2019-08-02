#include "rpn/fcos/fcos.h"
#include "rpn/utils.h"
#include "defaults.h"
#include <math.h>
#include <torch/torch.h>

namespace rcnn
{
namespace modeling
{

FCOSHeadImpl::FCOSHeadImpl(int64_t in_channels)
{
    auto num_classes = rcnn::config::GetCFG<int>({"MODEL", "FCOS", "NUM_CLASSES"}) - 1;
    int num_convs = rcnn::config::GetCFG<int>({"MODEL", "FCOS", "NUM_CONVS"});

    for (int i = 0; i < num_convs; ++i)
    {
        cls_tower->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).padding(1).stride(1)));
        cls_tower->push_back(torch::nn::Functional(torch::group_norm, 32, torch::ones({in_channels}), torch::ones({in_channels}), 1e-5, true));
        cls_tower->push_back(torch::nn::Functional(torch::relu));

        bbox_tower->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).padding(1).stride(1)));
        bbox_tower->push_back(torch::nn::Functional(torch::group_norm, 32, torch::ones({in_channels}), torch::ones({in_channels}), 1e-5, true));
        bbox_tower->push_back(torch::nn::Functional(torch::relu));
    }

    register_module("cls_tower", cls_tower);
    register_module("bbox_tower", bbox_tower);

    cls_logits = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, in_channels, 3).padding(1).stride(1));
    bbox_pred = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 4, 3).padding(1).stride(1));
    centerness = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 1, 3).padding(1).stride(1));

    //to make member function
    for (auto m : cls_tower->modules())
    {
        auto modules = m->modules();
        for (auto md : modules)
        {
            auto cv2 = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(md);
            if (cv2 != nullptr)
            {
                torch::nn::init::normal_(cv2->weight, 0, 0.01);
                torch::nn::init::normal_(cv2->bias, 0, 0.01);
            }
        }
    }

    for (auto m : bbox_tower->modules())
    {
        auto modules = m->modules();
        for (auto md : modules)
        {
            auto cv2 = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(md);
            if (cv2 != nullptr)
            {
                torch::nn::init::normal_(cv2->weight, 0, 0.01);
                torch::nn::init::normal_(cv2->bias, 0, 0.01);
            }
        }
    }

    auto prior_prob = rcnn::config::GetCFG<float>({"MODEL", "FCOS", "PRIOR_PROB"}); // cfg.MODEL.FCOS.PRIOR_PROB
    auto bias_value = -log((1 - prior_prob) / prior_prob);
    torch::nn::init::constant_(cls_logits->bias, bias_value);
    for (int i = 0; i < 5; i++)
        this->scales.push_back(layers::Scale(1.0));
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> FCOSHeadImpl::forward(std::vector<torch::Tensor> &x)
{
    std::vector<torch::Tensor> logits;
    std::vector<torch::Tensor> bbox_reg;
    std::vector<torch::Tensor> centerness;

    for (size_t i = 0; i < x.size(); ++i)
    {
        auto cls_t = this->cls_tower->forward(x[i]);
        logits.push_back(this->cls_logits->forward(cls_t));
        centerness.push_back(this->centerness->forward(cls_t));
        bbox_reg.push_back(torch::exp(scales[i].forward(this->bbox_pred->forward(this->bbox_tower->forward(x[i])))));
    }

    return std::tie(logits, bbox_reg, centerness);
}

FCOSModuleImpl::FCOSModuleImpl(int64_t in_channels)
{
    head = FCOSHead(in_channels);

    auto pre_nms_thresh = rcnn::config::GetCFG<float>({"MODEL", "FCOS", "INFERENCE_TH"});         //config.MODEL.FCOS.INFERENCE_TH
    auto pre_nms_top_n = rcnn::config::GetCFG<int>({"MODEL", "FCOS", "PRE_NMS_TOP_N"});           //config.MODEL.FCOS.PRE_NMS_TOP_N
    auto nms_thresh = rcnn::config::GetCFG<float>({"MODEL", "FCOS", "NMS_TH"});                   //config.MODEL.FCOS.NMS_TH
    auto fpn_post_nms_top_n = rcnn::config::GetCFG<int>({"MODEL", "FCOS", "DETECTIONS_PER_IMG"}); //config.TEST.DETECTIONS_PER_IMG
    auto numclass = rcnn::config::GetCFG<int>({"MODEL", "FCOS", "NUM_CLASSES"});

    box_selector_test = MakeFCOSPostprocessor(pre_nms_thresh, pre_nms_top_n, nms_thresh, fpn_post_nms_top_n, 0, numclass);

    //loss_evaluator = make_fcos_loss_evaluator(cfg)
    fpn_strides = rcnn::config::GetCFG<std::vector<int64_t>>({"MODEL", "FCOS", "FPN_STRIDES"});
}

std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>>
FCOSModuleImpl::forward(rcnn::structures::ImageList &images, std::vector<torch::Tensor> &features)
{
    std::vector<torch::Tensor> box_cls;
    std::vector<torch::Tensor> box_regression;
    std::vector<torch::Tensor> centerness;

    std::tie(box_cls, box_regression, centerness) = head->forward(features);
    auto locations = compute_locations(features);
    return this->_forward_test(locations, box_cls, box_regression, centerness, images.get_image_sizes());
}

std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>>
FCOSModuleImpl::forward(rcnn::structures::ImageList &images, std::vector<torch::Tensor> &features, std::vector<rcnn::structures::BoxList> targets)
{
    std::vector<torch::Tensor> box_cls;
    std::vector<torch::Tensor> box_regression;
    std::vector<torch::Tensor> centerness;

    std::tie(box_cls, box_regression, centerness) = head->forward(features);
    auto locations = compute_locations(features);
    return this->_forward_train(locations, box_cls, box_regression, centerness, targets);
}

std::vector<torch::Tensor> FCOSModuleImpl::compute_locations(std::vector<torch::Tensor> &features)
{
    std::vector<torch::Tensor> locations;
    for (size_t level = 0; level < features.size(); ++level)
    {
        auto feature = features[level];
        int h = feature.size(-2);
        int w = feature.size(-1);
        auto locations_per_level = this->compute_locations_per_level(
            h, w, this->fpn_strides[level],
            feature.device());
        locations.push_back(locations_per_level);
    }
    return locations;
}

torch::Tensor
FCOSModuleImpl::compute_locations_per_level(int64_t h, int64_t w, int stride, torch::Device device)
{
    auto op1 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto shifts_x = torch::arange(0, w * stride, stride, op1);
    auto shifts_y = torch::arange(0, h * stride, stride, op1);
    torch::Tensor shift_y, shift_x;
    auto x = torch::meshgrid({shifts_y, shifts_x});
    shift_y = x[0];
    shift_x = x[1];

    shift_x = shift_x.reshape(-1);
    shift_y = shift_y.reshape(-1);
    auto locations = torch::stack({shift_x, shift_y}, 1) + (int)(stride / 2);
    return locations;
}

std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>>
FCOSModuleImpl::_forward_test(const std::vector<torch::Tensor> &locations,
                              const std::vector<torch::Tensor> &box_cls,
                              const std::vector<torch::Tensor> &box_regression,
                              const std::vector<torch::Tensor> &centerness,
                              const std::vector<std::pair<int64_t, int64_t>> &image_sizes)
{

    auto t = box_selector_test->forward(locations, box_cls, box_regression, centerness, image_sizes);
    return std::make_pair(t, std::map<std::string, torch::Tensor>());
}

std::pair<std::vector<rcnn::structures::BoxList>, std::map<std::string, torch::Tensor>>
FCOSModuleImpl::_forward_train(const std::vector<torch::Tensor> &locations,
                               const std::vector<torch::Tensor> &box_cls,
                               const std::vector<torch::Tensor> &box_regression,
                               const std::vector<torch::Tensor> &centerness,
                               const std::vector<rcnn::structures::BoxList> &targets)
{
    return std::make_pair(std::vector<rcnn::structures::BoxList>(), std::map<std::string, torch::Tensor>());
}

} // namespace modeling
} // namespace rcnn