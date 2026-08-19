#include <gtest/gtest.h>
#include <microllm/base/device.h>
#include <microllm/base/dtype.h>
#include <microllm/core/storage.h>
#include <microllm/core/tensor.h>
#include <microllm/core/tensor_view.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>
#include <microllm/profiling/trace.h>
#include <microllm/ops/context.h>
#include <microllm/ops/ops.h>
#include <microllm/ops/low_level.h>
#include <microllm/autograd/autograd.h>
#include <microllm/training/optimizer.h>
#include <microllm/training/checkpoint.h>
#include <microllm/model/config.h>
#include <microllm/model/huggingface.h>
#include <microllm/model/model.h>
#include <microllm/io/byte_tokenizer.h>
#include <microllm/io/bpe_tokenizer.h>
#include <microllm/io/token_dataset.h>
#include <microllm/io/sft.h>
#include <microllm/io/safetensors.h>
#include <microllm/training/trainer.h>
#include <microllm/inference/kv_cache.h>
#include <microllm/inference/generator.h>
#if MICROLLM_HAS_RCCL
#include <microllm/multi_gpu/communicator.h>
#include <microllm/multi_gpu/gradient_bucket.h>
#include <microllm/multi_gpu/data_parallel.h>
#endif

TEST(PublicHeaders, CanBeIncludedTogether) { SUCCEED(); }
