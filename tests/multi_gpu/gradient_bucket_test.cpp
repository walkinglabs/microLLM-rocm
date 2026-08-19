#include <vector>

#include <gtest/gtest.h>
#include <microllm/model/model.h>
#include <microllm/multi_gpu/gradient_bucket.h>
#include <microllm/runtime/runtime.h>

namespace microllm::multi_gpu {

TEST(RcclGradientBucketTest, OneLargeBucketAveragesEveryTinyModelGradient) {
    if (runtime::hip_device_count() < 2) GTEST_SKIP() << "two visible HIP devices required";
    const model::ModelConfig config{.vocabulary_size = 8,
                                    .dimension = 8,
                                    .layers = 1,
                                    .heads = 2,
                                    .kv_heads = 1,
                                    .ffn_dimension = 16,
                                    .max_sequence_length = 4,
                                    .rope_base = 10000.0F,
                                    .tie_embeddings = false};
    model::TransformerModel rank0(config, 97);
    model::TransformerModel rank1(config, 97);
    rank0.to(Device::hip(0));
    rank1.to(Device::hip(1));
    const auto input0 = Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4});
    const auto input1 = Tensor::from_int32_vector({3, 2, 1, 0}, {1, 4});
    const auto target = Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4});
    rank0.loss(input0, target).backward();
    rank1.loss(input1, target).backward();
    Communicator communicator({0, 1});
    const auto parameters0 = rank0.parameters();
    const auto parameters1 = rank1.parameters();
    const auto stats = all_reduce_gradients(communicator, {parameters0, parameters1},
                                            1024 * 1024);
    EXPECT_EQ(stats.bucket_count, 1U);
    EXPECT_EQ(stats.parameter_count, parameters0.size());
    for (std::size_t index = 0; index < parameters0.size(); ++index) {
        EXPECT_EQ(parameters0[index]->grad().to_vector(), parameters1[index]->grad().to_vector());
    }
}

}  // namespace microllm::multi_gpu
