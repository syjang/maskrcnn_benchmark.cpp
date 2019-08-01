#pragma once
#include <torch/torch.h>
#include "bounding_box.h"
#include "box_coder.h"

namespace rcnn
{
namespace modeling
{

class FCOSPostProcessorImpl : public torch::nn::Module
{
public:
    FCOSPostProcessorImpl(float pre_nms_thresh, int pre_nms_top_n,
                          float nms_thresh, int fpn_post_num_top_n, int min_size,
                          int num_classes);

    std::vector<structures::BoxList> forward(std::vector<torch::Tensor> &locations,
                                             std::vector<torch::Tensor> &box_cls,
                                             std::vector<torch::Tensor> &box_regression,
                                             std::vector<torch::Tensor> &centerness,
                                             std::vector<std::pair<int64_t, int64_t>> image_sizes);

private:
    std::vector<structures::BoxList> forward_for_single_feature_map(torch::Tensor &locations,
                                                                    torch::Tensor &box_cls,
                                                                    torch::Tensor &box_regression,
                                                                    torch::Tensor &centerness,
                                                                    const std::vector<std::pair<int64_t, int64_t>> &image_sizes);

    std::vector<structures::BoxList> select_over_all_levels(const std::vector<structures::BoxList> &boxlists);

    float pre_nms_thresh;
    int pre_nms_top_n;
    float nms_thresh;
    int fpn_post_num_top_n;
    int min_size;
    int num_classes;
};

TORCH_MODULE(FCOSPostProcessor);

FCOSPostProcessor MakeFCOSPostprocessor(float pre_nms_thresh, int pre_nms_top_n,
                                        float nms_thresh, int fpn_post_num_top_n, int min_size,
                                        int num_classes);
} // namespace modeling
} // namespace rcnn