#pragma once

#include <microllm/core/tensor_view.h>
#include <microllm/ops/context.h>

namespace microllm::ops {

void add_out(TensorView output, ConstTensorView left, ConstTensorView right,
             const OpContext& context = {});
void multiply_out(TensorView output, ConstTensorView left, ConstTensorView right,
                  const OpContext& context = {});

}  // namespace microllm::ops
