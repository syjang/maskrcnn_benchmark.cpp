#include "rpn/fcos/loss.h"

//todo make it
namespace rcnn
{
namespace modeling
{
FCOSLossComputation::FCOSLossComputation()
{
    // centerness_loss_func = torch::nn::Functional(torch::binary_cross_entropy_with_logits);
}

void FCOSLossComputation::prepare_targtes() {}
void FCOSLossComputation::compute_targets_for_locations() {}
void FCOSLossComputation::compute_centerness_targets() {}

FCOSLossComputation make_fcos_loss_evaluator()
{
    return FCOSLossComputation();
}
} // namespace modeling
} // namespace rcnn