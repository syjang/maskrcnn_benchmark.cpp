#include "scale.h"

namespace rcnn
{
namespace layers
{
Scale::Scale(float init_value) : scale(init_value)
{
}

torch::Tensor Scale::forward(torch::Tensor input)
{
    return input * this->scale;
}
} // namespace layers
} // namespace rcnn