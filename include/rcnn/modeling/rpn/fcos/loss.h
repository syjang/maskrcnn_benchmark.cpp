#pragma once
#include <torch/torch.h>
#include "matcher.h"
#include "balanced_positive_negative_sampler.h"
#include "box_coder.h"
#include "bounding_box.h"
#include <set>
#include "iou_loss.h"

//when inferencing , loss cumpute is not required

namespace rcnn
{
namespace modeling
{
class FCOSLossComputation
{
public:
    FCOSLossComputation();

    void prepare_targtes();
    void compute_targets_for_locations();
    void compute_centerness_targets();

private:
    layers::IOULoss box_reg_loss_func = nullptr;
    // torch::nn::Functional centerness_loss_func = nullptr;
};

FCOSLossComputation make_fcos_loss_evaluator();

} // namespace modeling
} // namespace rcnn