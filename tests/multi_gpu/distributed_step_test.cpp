#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/model/model.h>
#include <microllm/multi_gpu/communicator.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/optimizer.h>

namespace microllm::multi_gpu {
namespace {

model::ModelConfig distributed_config() {
    return {.vocabulary_size = 8,
            .dimension = 8,
            .layers = 1,
            .heads = 2,
            .kv_heads = 1,
            .ffn_dimension = 16,
            .max_sequence_length = 4,
            .rope_base = 10000.0F,
            .tie_embeddings = false};
}

}  // namespace

TEST(RcclDistributedStepTest, TwoRanksMatchSingleGlobalBatchUpdate) {
    if (runtime::hip_device_count() < 2) GTEST_SKIP() << "two visible HIP devices required";
    const auto config = distributed_config();
    const training::AdamWConfig optimizer_config{.learning_rate = 0.005F,
                                                  .beta1 = 0.9F,
                                                  .beta2 = 0.99F,
                                                  .epsilon = 1.0e-8F,
                                                  .weight_decay = 0.0F};
    model::TransformerModel reference(config, 83);
    model::TransformerModel rank0(config, 83);
    model::TransformerModel rank1(config, 83);
    rank0.to(Device::hip(0));
    rank1.to(Device::hip(1));
    training::AdamW reference_optimizer(reference.parameters(), optimizer_config);
    training::AdamW rank0_optimizer(rank0.parameters(), optimizer_config);
    training::AdamW rank1_optimizer(rank1.parameters(), optimizer_config);

    const auto input0 = Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4});
    const auto target0 = Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4});
    const auto input1 = Tensor::from_int32_vector({3, 2, 1, 0}, {1, 4});
    const auto target1 = Tensor::from_int32_vector({2, 1, 0, 3}, {1, 4});
    const auto global_input = Tensor::from_int32_vector({0, 1, 2, 3, 3, 2, 1, 0}, {2, 4});
    const auto global_target = Tensor::from_int32_vector({1, 2, 3, 0, 2, 1, 0, 3}, {2, 4});

    reference.loss(global_input, global_target).backward();
    rank0.loss(input0, target0).backward();
    rank1.loss(input1, target1).backward();

    Communicator communicator({0, 1});
    const auto reference_parameters = reference.parameters();
    const auto rank0_parameters = rank0.parameters();
    const auto rank1_parameters = rank1.parameters();
    ASSERT_EQ(rank0_parameters.size(), rank1_parameters.size());
    for (std::size_t index = 0; index < rank0_parameters.size(); ++index) {
        std::vector<Tensor> gradients{rank0_parameters[index]->grad(),
                                      rank1_parameters[index]->grad()};
        communicator.all_reduce(gradients, true);
        rank0_parameters[index]->set_grad(gradients[0]);
        rank1_parameters[index]->set_grad(gradients[1]);
    }
    reference_optimizer.step();
    rank0_optimizer.step();
    rank1_optimizer.step();

    float rank_difference = 0.0F;
    float reference_difference = 0.0F;
    for (std::size_t index = 0; index < rank0_parameters.size(); ++index) {
        const auto reference_values = reference_parameters[index]->data().to_vector();
        const auto rank0_values = rank0_parameters[index]->data().to_vector();
        const auto rank1_values = rank1_parameters[index]->data().to_vector();
        for (std::size_t value = 0; value < rank0_values.size(); ++value) {
            rank_difference = std::max(rank_difference,
                                       std::abs(rank0_values[value] - rank1_values[value]));
            reference_difference = std::max(
                reference_difference, std::abs(reference_values[value] - rank0_values[value]));
        }
    }
    std::cout << "rank_parameter_max_difference=" << rank_difference << '\n';
    std::cout << "single_vs_two_rank_max_difference=" << reference_difference << '\n';
    EXPECT_EQ(rank_difference, 0.0F);
    EXPECT_LE(reference_difference, 2.0e-5F);
}

}  // namespace microllm::multi_gpu
