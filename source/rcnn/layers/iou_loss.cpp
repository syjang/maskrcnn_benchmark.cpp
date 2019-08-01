#include "iou_loss.h"

namespace rcnn
{
namespace layers
{
torch::Tensor IOULoss::forward(torch::Tensor pred, torch::Tensor target, torch::Tensor weight)
{
    return torch::Tensor();
}
} // namespace layers
} // namespace rcnn