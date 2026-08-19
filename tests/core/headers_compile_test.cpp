#include <gtest/gtest.h>
#include <microllm/base/device.h>
#include <microllm/base/dtype.h>
#include <microllm/core/storage.h>
#include <microllm/core/tensor.h>
#include <microllm/core/tensor_view.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>
#include <microllm/ops/context.h>
#include <microllm/ops/ops.h>
#include <microllm/autograd/autograd.h>

TEST(PublicHeaders, CanBeIncludedTogether) { SUCCEED(); }
