#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/model/model.h>
#include <microllm/multi_gpu/gradient_bucket.h>
#include <microllm/runtime/memory.h>
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
    EXPECT_EQ(stats.bucket_tensor_count, 2U);
    EXPECT_EQ(stats.average_tensor_count, 0U);
    EXPECT_EQ(stats.unpacked_tensor_count, parameters0.size() * 2U);
    EXPECT_EQ(stats.pack_copy_calls, parameters0.size() * 2U);
    EXPECT_EQ(stats.unpack_copy_calls, parameters0.size() * 2U);
    EXPECT_EQ(stats.temporary_elements, stats.total_elements * 4U);
    EXPECT_EQ(stats.temporary_bytes, stats.temporary_elements * sizeof(float));
    for (std::size_t index = 0; index < parameters0.size(); ++index) {
        EXPECT_EQ(parameters0[index]->grad().to_vector(), parameters1[index]->grad().to_vector());
    }
}

TEST(RcclGradientBucketTest, PersistentPlanReusesEveryReducerStorageAddress) {
    static_assert(std::is_move_constructible_v<GradientBucketPlan>);
    static_assert(std::is_move_assignable_v<GradientBucketPlan>);
    static_assert(!std::is_copy_constructible_v<GradientBucketPlan>);
    static_assert(!std::is_copy_assignable_v<GradientBucketPlan>);
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
    model::TransformerModel rank0(config, 101);
    model::TransformerModel rank1(config, 101);
    rank0.to(Device::hip(0));
    rank1.to(Device::hip(1));
    const auto input0 = Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4});
    const auto input1 = Tensor::from_int32_vector({3, 2, 1, 0}, {1, 4});
    const auto target = Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4});
    auto parameters0 = rank0.parameters();
    auto parameters1 = rank1.parameters();
    const auto backward = [&] {
        for (auto* parameter : parameters0) parameter->zero_grad();
        for (auto* parameter : parameters1) parameter->zero_grad();
        rank0.loss(input0, target).backward();
        rank1.loss(input1, target).backward();
    };

    Communicator communicator({0, 1});
    GradientBucketPlan plan;
    EXPECT_FALSE(plan.initialized());
    backward();
    const auto allocation_before = runtime::allocation_stats(Device::hip(0));
    const auto first = all_reduce_gradients(
        communicator, {parameters0, parameters1}, 1024 * 1024, true, &plan);
    const auto allocation_after = runtime::allocation_stats(Device::hip(0));
    EXPECT_TRUE(plan.initialized());
    EXPECT_TRUE(first.persistent_storage);
    EXPECT_FALSE(first.plan_reused);
    EXPECT_EQ(first.temporary_elements, 0U);
    EXPECT_EQ(first.temporary_bytes, 0U);
    EXPECT_EQ(first.plan_capacity_elements, first.total_elements * 4U);
    EXPECT_EQ(first.plan_capacity_bytes,
              first.plan_capacity_elements * sizeof(float));
    EXPECT_EQ(allocation_after.allocation_calls - allocation_before.allocation_calls,
              first.bucket_tensor_count + first.unpacked_tensor_count);

    std::vector<const void*> gradient_addresses;
    gradient_addresses.reserve(parameters0.size() + parameters1.size());
    for (const auto* parameter : parameters0) {
        gradient_addresses.push_back(parameter->grad().data());
    }
    for (const auto* parameter : parameters1) {
        gradient_addresses.push_back(parameter->grad().data());
    }

    backward();
    const auto reuse_before = runtime::allocation_stats(Device::hip(0));
    const auto second = all_reduce_gradients(
        communicator, {parameters0, parameters1}, 1024 * 1024, true, &plan);
    const auto reuse_after = runtime::allocation_stats(Device::hip(0));
    EXPECT_TRUE(second.persistent_storage);
    EXPECT_TRUE(second.plan_reused);
    EXPECT_EQ(second.plan_capacity_bytes, first.plan_capacity_bytes);
    EXPECT_EQ(reuse_after.allocation_calls - reuse_before.allocation_calls, 0U);
    std::size_t address = 0;
    for (const auto* parameter : parameters0) {
        EXPECT_EQ(parameter->grad().data(), gradient_addresses[address++]);
    }
    for (const auto* parameter : parameters1) {
        EXPECT_EQ(parameter->grad().data(), gradient_addresses[address++]);
    }
    for (std::size_t index = 0; index < parameters0.size(); ++index) {
        EXPECT_EQ(parameters0[index]->grad().to_vector(),
                  parameters1[index]->grad().to_vector());
    }
    EXPECT_THROW(
        (void)all_reduce_gradients(
            communicator, {parameters0, parameters1}, 512 * 1024, true, &plan),
        std::invalid_argument);
    EXPECT_THROW(
        (void)all_reduce_gradients(
            communicator, {parameters0, parameters1}, 1024 * 1024, true,
            &plan, true),
        std::invalid_argument);

    GradientBucketPlan moved = std::move(plan);
    EXPECT_FALSE(plan.initialized());
    EXPECT_TRUE(moved.initialized());
    moved.clear();
    EXPECT_FALSE(moved.initialized());
}

TEST(RcclGradientBucketTest, GradientViewsShareBucketStorageAndSkipUnpackCopies) {
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
    model::TransformerModel rank0(config, 107);
    model::TransformerModel rank1(config, 107);
    rank0.to(Device::hip(0));
    rank1.to(Device::hip(1));
    const auto input0 = Tensor::from_int32_vector({0, 1, 2, 3}, {1, 4});
    const auto input1 = Tensor::from_int32_vector({3, 2, 1, 0}, {1, 4});
    const auto target = Tensor::from_int32_vector({1, 2, 3, 0}, {1, 4});
    auto parameters0 = rank0.parameters();
    auto parameters1 = rank1.parameters();
    const auto backward = [&] {
        for (auto* parameter : parameters0) parameter->zero_grad();
        for (auto* parameter : parameters1) parameter->zero_grad();
        rank0.loss(input0, target).backward();
        rank1.loss(input1, target).backward();
    };

    Communicator communicator({0, 1});
    GradientBucketPlan plan;
    backward();
    const auto allocation_before = runtime::allocation_stats(Device::hip(0));
    const auto first = all_reduce_gradients(
        communicator, {parameters0, parameters1}, 1024 * 1024, true,
        &plan, true);
    const auto allocation_after = runtime::allocation_stats(Device::hip(0));
    EXPECT_TRUE(first.persistent_storage);
    EXPECT_EQ(first.bucket_count, 1U);
    EXPECT_EQ(first.bucket_tensor_count, 2U);
    EXPECT_EQ(first.unpacked_tensor_count, 0U);
    EXPECT_EQ(first.gradient_view_count, parameters0.size() * 2U);
    EXPECT_EQ(first.unpack_copy_calls, 0U);
    EXPECT_EQ(first.plan_capacity_elements, first.total_elements * 2U);
    EXPECT_EQ(allocation_after.allocation_calls - allocation_before.allocation_calls,
              first.bucket_tensor_count);

    std::vector<const void*> addresses;
    addresses.reserve(parameters0.size() + parameters1.size());
    for (const auto& parameters : {parameters0, parameters1}) {
        const auto* bucket_storage = parameters.front()->grad().storage().data();
        std::int64_t expected_offset = 0;
        for (const auto* parameter : parameters) {
            EXPECT_EQ(parameter->grad().storage().data(), bucket_storage);
            EXPECT_EQ(parameter->grad().storage_offset(), expected_offset);
            EXPECT_TRUE(parameter->grad().is_contiguous());
            addresses.push_back(parameter->grad().data());
            expected_offset += parameter->grad().numel();
        }
    }
    for (std::size_t index = 0; index < parameters0.size(); ++index) {
        EXPECT_EQ(parameters0[index]->grad().to_vector(),
                  parameters1[index]->grad().to_vector());
    }

    backward();
    const auto reuse_before = runtime::allocation_stats(Device::hip(0));
    const auto second = all_reduce_gradients(
        communicator, {parameters0, parameters1}, 1024 * 1024, true,
        &plan, true);
    const auto reuse_after = runtime::allocation_stats(Device::hip(0));
    EXPECT_TRUE(second.plan_reused);
    EXPECT_EQ(second.unpack_copy_calls, 0U);
    EXPECT_EQ(reuse_after.allocation_calls - reuse_before.allocation_calls, 0U);
    std::size_t address = 0;
    for (const auto& parameters : {parameters0, parameters1}) {
        for (const auto* parameter : parameters) {
            EXPECT_EQ(parameter->grad().data(), addresses[address++]);
        }
    }

}

TEST(RcclRankGradientBucketTest, WorldOneBucketPreservesEveryGradient) {
    if (runtime::hip_device_count() < 1) GTEST_SKIP() << "visible HIP device required";
    const auto id = create_communicator_id();
    RankCommunicator communicator(0, 1, 0, id);
    autograd::Value first(Tensor({2}, DType::Float32, Device::hip(0)), true);
    autograd::Value second(Tensor({3}, DType::Float32, Device::hip(0)), true);
    first.set_grad(Tensor::from_vector({1, 2}, {2}).to(Device::hip(0)));
    second.set_grad(Tensor::from_vector({3, 4, 5}, {3}).to(Device::hip(0)));
    const auto stats = all_reduce_rank_gradients(
        communicator, {&first, &second}, 4096);
    EXPECT_EQ(stats.bucket_count, 1U);
    EXPECT_EQ(stats.parameter_count, 2U);
    EXPECT_EQ(stats.total_elements, 5U);
    EXPECT_EQ(stats.pack_copy_calls, 2U);
    EXPECT_EQ(stats.unpack_copy_calls, 2U);
    EXPECT_EQ(first.grad().to_vector(), (std::vector<float>{1, 2}));
    EXPECT_EQ(second.grad().to_vector(), (std::vector<float>{3, 4, 5}));
    EXPECT_THROW(
        (void)all_reduce_rank_gradients(communicator, {&first}, 1),
        std::invalid_argument);
}

TEST(RcclRankGradientBucketTest,
     PersistentPlanReusesRankLocalStorageAndRejectsContractChanges) {
    static_assert(std::is_move_constructible_v<RankGradientBucketPlan>);
    static_assert(std::is_move_assignable_v<RankGradientBucketPlan>);
    static_assert(!std::is_copy_constructible_v<RankGradientBucketPlan>);
    static_assert(!std::is_copy_assignable_v<RankGradientBucketPlan>);
    if (runtime::hip_device_count() < 1) {
        GTEST_SKIP() << "visible HIP device required";
    }
    const auto id = create_communicator_id();
    RankCommunicator communicator(0, 1, 0, id);
    autograd::Value first(Tensor({2}, DType::Float32, Device::hip(0)), true);
    autograd::Value second(Tensor({3}, DType::Float32, Device::hip(0)), true);
    first.set_grad(Tensor::from_vector({1, 2}, {2}).to(Device::hip(0)));
    second.set_grad(Tensor::from_vector({3, 4, 5}, {3}).to(Device::hip(0)));

    RankGradientBucketPlan plan;
    EXPECT_FALSE(plan.initialized());
    const auto first_before = runtime::allocation_stats(Device::hip(0));
    const auto first_stats = all_reduce_rank_gradients(
        communicator, {&first, &second}, 4096, &plan);
    const auto first_after = runtime::allocation_stats(Device::hip(0));
    EXPECT_TRUE(plan.initialized());
    EXPECT_TRUE(first_stats.persistent_storage);
    EXPECT_FALSE(first_stats.plan_reused);
    EXPECT_EQ(first_stats.bucket_count, 1U);
    EXPECT_EQ(first_stats.plan_capacity_elements, 10U);
    EXPECT_EQ(first_stats.plan_capacity_bytes, 10U * sizeof(float));
    EXPECT_EQ(first_after.allocation_calls - first_before.allocation_calls, 3U);
    const auto* first_address = first.grad().data();
    const auto* second_address = second.grad().data();

    first.set_grad(Tensor::from_vector({6, 7}, {2}).to(Device::hip(0)));
    second.set_grad(Tensor::from_vector({8, 9, 10}, {3}).to(Device::hip(0)));
    const auto reuse_before = runtime::allocation_stats(Device::hip(0));
    const auto reused = all_reduce_rank_gradients(
        communicator, {&first, &second}, 4096, &plan);
    const auto reuse_after = runtime::allocation_stats(Device::hip(0));
    EXPECT_TRUE(reused.persistent_storage);
    EXPECT_TRUE(reused.plan_reused);
    EXPECT_EQ(reuse_after.allocation_calls - reuse_before.allocation_calls, 0U);
    EXPECT_EQ(first.grad().data(), first_address);
    EXPECT_EQ(second.grad().data(), second_address);
    EXPECT_EQ(first.grad().to_vector(), (std::vector<float>{6, 7}));
    EXPECT_EQ(second.grad().to_vector(), (std::vector<float>{8, 9, 10}));
    EXPECT_THROW(
        (void)all_reduce_rank_gradients(
            communicator, {&first, &second}, sizeof(float), &plan),
        std::invalid_argument);

    RankGradientBucketPlan moved = std::move(plan);
    EXPECT_FALSE(plan.initialized());
    EXPECT_TRUE(moved.initialized());
    moved.clear();
    EXPECT_FALSE(moved.initialized());
}

TEST(RcclRankGradientBucketTest,
     GradientViewsSharePersistentRankBucketAndSkipUnpackCopies) {
    if (runtime::hip_device_count() < 1) {
        GTEST_SKIP() << "visible HIP device required";
    }
    const auto id = create_communicator_id();
    RankCommunicator communicator(0, 1, 0, id);
    autograd::Value first(Tensor({2}, DType::Float32, Device::hip(0)), true);
    autograd::Value second(Tensor({3}, DType::Float32, Device::hip(0)), true);
    first.set_grad(Tensor::from_vector({1, 2}, {2}).to(Device::hip(0)));
    second.set_grad(Tensor::from_vector({3, 4, 5}, {3}).to(Device::hip(0)));

    RankGradientBucketPlan plan;
    const auto first_before = runtime::allocation_stats(Device::hip(0));
    const auto first_stats = all_reduce_rank_gradients(
        communicator, {&first, &second}, 4096, &plan, true);
    const auto first_after = runtime::allocation_stats(Device::hip(0));
    EXPECT_TRUE(first_stats.persistent_storage);
    EXPECT_FALSE(first_stats.plan_reused);
    EXPECT_EQ(first_stats.gradient_view_count, 2U);
    EXPECT_EQ(first_stats.unpack_copy_calls, 0U);
    EXPECT_EQ(first_stats.plan_capacity_elements, 5U);
    EXPECT_EQ(first_stats.plan_capacity_bytes, 5U * sizeof(float));
    EXPECT_EQ(first_after.allocation_calls - first_before.allocation_calls, 1U);
    EXPECT_EQ(first.grad().storage().data(), second.grad().storage().data());
    EXPECT_EQ(first.grad().storage_offset(), 0);
    EXPECT_EQ(second.grad().storage_offset(), 2);
    const auto* first_address = first.grad().data();
    const auto* second_address = second.grad().data();

    first.set_grad(Tensor::from_vector({6, 7}, {2}).to(Device::hip(0)));
    second.set_grad(Tensor::from_vector({8, 9, 10}, {3}).to(Device::hip(0)));
    const auto reuse_before = runtime::allocation_stats(Device::hip(0));
    const auto reused = all_reduce_rank_gradients(
        communicator, {&first, &second}, 4096, &plan, true);
    const auto reuse_after = runtime::allocation_stats(Device::hip(0));
    EXPECT_TRUE(reused.plan_reused);
    EXPECT_EQ(reused.gradient_view_count, 2U);
    EXPECT_EQ(reused.unpack_copy_calls, 0U);
    EXPECT_EQ(reuse_after.allocation_calls - reuse_before.allocation_calls, 0U);
    EXPECT_EQ(first.grad().data(), first_address);
    EXPECT_EQ(second.grad().data(), second_address);
    EXPECT_EQ(first.grad().to_vector(), (std::vector<float>{6, 7}));
    EXPECT_EQ(second.grad().to_vector(), (std::vector<float>{8, 9, 10}));
    EXPECT_THROW(
        (void)all_reduce_rank_gradients(
            communicator, {&first, &second}, 4096, &plan, false),
        std::invalid_argument);
    EXPECT_THROW(
        (void)all_reduce_rank_gradients(
            communicator, {&first, &second}, 4096, nullptr, true),
        std::invalid_argument);
}

}  // namespace microllm::multi_gpu
