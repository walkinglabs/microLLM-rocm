#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/autograd/autograd.h>
#include <microllm/base/low_precision.h>
#include <microllm/ops/ops.h>
#include <microllm/training/optimizer.h>

namespace microllm::training {
using namespace microllm::autograd;

TEST(SgdTest, ReducesAQuadraticLoss) {
    Value parameter(Tensor::from_vector({2.0F}, {1}), true);
    SGD optimizer({&parameter}, 0.1F);
    const auto before = sum(multiply(parameter, parameter)).data().to_vector()[0];
    sum(multiply(parameter, parameter)).backward();
    optimizer.step();
    optimizer.zero_grad();
    const auto after = sum(multiply(parameter, parameter)).data().to_vector()[0];
    EXPECT_LT(after, before);
    EXPECT_FALSE(parameter.has_grad());
}

TEST(AdamWTest, FirstStepMatchesBiasCorrectedHandCalculation) {
    Value parameter(Tensor::from_vector({2.0F, -3.0F}, {2}), true);
    const AdamWConfig config{.learning_rate = 0.1F,
                             .beta1 = 0.9F,
                             .beta2 = 0.999F,
                             .epsilon = 1.0e-8F,
                             .weight_decay = 0.1F};
    AdamW optimizer({&parameter}, config);
    sum(scale(parameter, 2.0F)).backward();
    optimizer.step();
    const auto values = parameter.data().to_vector();
    EXPECT_NEAR(values[0], 2.0F * 0.99F - 0.1F, 1.0e-5F);
    EXPECT_NEAR(values[1], -3.0F * 0.99F - 0.1F, 1.0e-5F);
    EXPECT_EQ(optimizer.state().step, 1U);
    EXPECT_EQ(optimizer.step_count(), 1U);
}

TEST(AdamWTest, RestoredStateProducesTheSameNextStep) {
    const AdamWConfig config{.learning_rate = 0.01F, .weight_decay = 0.0F};
    Value uninterrupted(Tensor::from_vector({1.0F, -2.0F}, {2}), true);
    AdamW first({&uninterrupted}, config);
    sum(scale(uninterrupted, 3.0F)).backward();
    first.step();
    first.zero_grad();

    Value restored(Tensor::from_vector(uninterrupted.data().to_vector(), {2}), true);
    AdamW second({&restored}, config);
    second.load_state(first.state());

    sum(scale(uninterrupted, -2.0F)).backward();
    sum(scale(restored, -2.0F)).backward();
    first.step();
    second.step();
    EXPECT_EQ(restored.data().to_vector(), uninterrupted.data().to_vector());
    EXPECT_EQ(second.state().step, first.state().step);
}

TEST(AdamWTest, FusedBf16MirrorTracksUpdatedFp32Master) {
    Value parameter(Tensor::from_vector({1.0F, -2.0F, 3.0F}, {3}), true);
    auto mirror = parameter.data().cast(DType::BFloat16);
    AdamW optimizer({&parameter}, {.learning_rate = 0.01F, .weight_decay = 0.0F},
                    {{&parameter, &mirror}});
    sum(scale(parameter, 2.0F)).backward();
    optimizer.step();
    EXPECT_EQ(mirror.dtype(), DType::BFloat16);
    EXPECT_EQ(mirror.cast(DType::Float32).to_vector(),
              parameter.data().cast(DType::BFloat16).cast(DType::Float32).to_vector());

    Tensor bad({2}, DType::BFloat16);
    EXPECT_THROW((void)AdamW({&parameter}, {}, {{&parameter, &bad}}),
                 std::invalid_argument);
}

TEST(AdamWTest, Bf16MomentPrimitiveMatchesRoundedStateReference) {
    Tensor parameter = Tensor::from_vector({1.0F, -2.0F, 0.5F}, {3});
    const auto gradient = Tensor::from_vector({0.5F, -0.25F, 0.125F}, {3});
    Tensor first({3}, DType::BFloat16);
    Tensor second({3}, DType::BFloat16);
    Tensor mirror({3}, DType::BFloat16);
    ops::fill_(first, 0.0F);
    ops::fill_(second, 0.0F);
    std::vector<float> expected_parameter{1.0F, -2.0F, 0.5F};
    std::vector<float> expected_first(3, 0.0F);
    std::vector<float> expected_second(3, 0.0F);
    const auto gradient_values = gradient.to_vector();
    for (int step = 1; step <= 2; ++step) {
        const auto first_correction =
            1.0F - std::pow(0.9F, static_cast<float>(step));
        const auto second_correction =
            1.0F - std::pow(0.99F, static_cast<float>(step));
        ops::adamw_update_bf16_moments_(
            parameter, gradient, first, second, &mirror, 0.01F, 0.9F,
            0.99F, 1.0e-8F, 0.1F, first_correction, second_correction);
        for (std::size_t index = 0; index < expected_parameter.size(); ++index) {
            expected_first[index] = static_cast<float>(BFloat16(
                0.9F * expected_first[index] +
                0.1F * gradient_values[index]));
            expected_second[index] = static_cast<float>(BFloat16(
                0.99F * expected_second[index] +
                0.01F * gradient_values[index] * gradient_values[index]));
            expected_parameter[index] *= 0.999F;
            expected_parameter[index] -=
                0.01F * (expected_first[index] / first_correction) /
                (std::sqrt(expected_second[index] / second_correction) +
                 1.0e-8F);
        }
    }
    EXPECT_EQ(parameter.to_vector(), expected_parameter);
    EXPECT_EQ(first.to_vector(), expected_first);
    EXPECT_EQ(second.to_vector(), expected_second);
    EXPECT_EQ(mirror.to_vector(),
              parameter.cast(DType::BFloat16).to_vector());
    Tensor bad_first({3}, DType::Float32);
    EXPECT_THROW(
        ops::adamw_update_bf16_moments_(
            parameter, gradient, bad_first, second, nullptr, 0.01F, 0.9F,
            0.99F, 1.0e-8F, 0.1F, 0.1F, 0.01F),
        std::invalid_argument);
}

TEST(AdamWTest, Bf16MomentStateIsCompactCanonicalAndExactlyRestorable) {
    const AdamWConfig bf16_config{
        .learning_rate = 0.01F,
        .beta1 = 0.9F,
        .beta2 = 0.99F,
        .epsilon = 1.0e-8F,
        .weight_decay = 0.0F,
        .moment_precision = AdamWConfig::MomentPrecision::BFloat16};
    Value uninterrupted(Tensor::from_vector({1.0F, -2.0F, 0.5F}, {3}), true);
    auto uninterrupted_mirror = uninterrupted.data().cast(DType::BFloat16);
    AdamW optimizer({&uninterrupted}, bf16_config,
                    {{&uninterrupted, &uninterrupted_mirror}});
    EXPECT_EQ(optimizer.moment_state_bytes(), 12U);
    sum(scale(uninterrupted, 0.5F)).backward();
    optimizer.step();
    optimizer.zero_grad();

    const auto canonical = optimizer.state();
    ASSERT_EQ(canonical.first_moments.size(), 1U);
    ASSERT_EQ(canonical.second_moments.size(), 1U);
    EXPECT_EQ(canonical.first_moments[0].dtype(), DType::Float32);
    EXPECT_EQ(canonical.second_moments[0].dtype(), DType::Float32);
    EXPECT_TRUE(canonical.first_moments[0].device().is_cpu());
    EXPECT_TRUE(canonical.second_moments[0].device().is_cpu());

    Value resumed(
        Tensor::from_vector(uninterrupted.data().to_vector(), {3}), true);
    auto resumed_mirror = resumed.data().cast(DType::BFloat16);
    AdamW resumed_optimizer({&resumed}, bf16_config,
                            {{&resumed, &resumed_mirror}});
    ops::fill_(resumed_mirror, 99.0F);
    resumed_optimizer.load_state(canonical);
    EXPECT_EQ(resumed_optimizer.moment_state_bytes(), 12U);
    EXPECT_EQ(resumed_mirror.to_vector(), uninterrupted_mirror.to_vector());

    sum(scale(uninterrupted, -0.25F)).backward();
    sum(scale(resumed, -0.25F)).backward();
    optimizer.step();
    resumed_optimizer.step();
    EXPECT_EQ(resumed.data().to_vector(), uninterrupted.data().to_vector());
    EXPECT_EQ(resumed_optimizer.state().first_moments[0].to_vector(),
              optimizer.state().first_moments[0].to_vector());
    EXPECT_EQ(resumed_optimizer.state().second_moments[0].to_vector(),
              optimizer.state().second_moments[0].to_vector());
    EXPECT_EQ(resumed_mirror.to_vector(), uninterrupted_mirror.to_vector());

    const AdamWConfig fp32_config{.moment_precision =
                                      AdamWConfig::MomentPrecision::Float32};
    Value fp32_parameter(Tensor::from_vector({1.0F, 2.0F, 3.0F}, {3}), true);
    AdamW fp32_optimizer({&fp32_parameter}, fp32_config);
    EXPECT_EQ(fp32_optimizer.moment_state_bytes(), 24U);
    EXPECT_THROW(
        (void)AdamW({&fp32_parameter}, bf16_config, {},
                    ops::AdamWImplementation::Vectorized),
        std::invalid_argument);
    EXPECT_THROW(
        (void)AdamW({&fp32_parameter}, bf16_config, {},
                    ops::AdamWImplementation::Auto, -2),
        std::invalid_argument);
    EXPECT_THROW(
        (void)AdamW({&fp32_parameter}, fp32_config, {},
                    ops::AdamWImplementation::Auto, 16),
        std::invalid_argument);
}

TEST(AdamWTest, Bf16MomentOptimizerMatchesRoundedReferenceForOneHundredSteps) {
    const AdamWConfig config{
        .learning_rate = 0.003F,
        .beta1 = 0.9F,
        .beta2 = 0.99F,
        .epsilon = 1.0e-8F,
        .weight_decay = 0.01F,
        .moment_precision = AdamWConfig::MomentPrecision::BFloat16};
    Value parameter(Tensor::from_vector({1.0F, -2.0F}, {2}), true);
    AdamW optimizer({&parameter}, config);
    std::vector<float> expected_parameter{1.0F, -2.0F};
    std::vector<float> expected_first(2, 0.0F);
    std::vector<float> expected_second(2, 0.0F);
    for (int step = 1; step <= 100; ++step) {
        const std::vector<float> gradients{
            static_cast<float>(step % 7 - 3) * 0.125F,
            static_cast<float>(step % 5 - 2) * -0.25F};
        parameter.set_grad(Tensor::from_vector(gradients, {2}));
        optimizer.step();
        const auto first_correction =
            1.0F - std::pow(0.9F, static_cast<float>(step));
        const auto second_correction =
            1.0F - std::pow(0.99F, static_cast<float>(step));
        for (std::size_t index = 0; index < gradients.size(); ++index) {
            expected_first[index] = static_cast<float>(BFloat16(
                config.beta1 * expected_first[index] +
                (1.0F - config.beta1) * gradients[index]));
            expected_second[index] = static_cast<float>(BFloat16(
                config.beta2 * expected_second[index] +
                (1.0F - config.beta2) * gradients[index] * gradients[index]));
            expected_parameter[index] *=
                1.0F - config.learning_rate * config.weight_decay;
            expected_parameter[index] -=
                config.learning_rate *
                (expected_first[index] / first_correction) /
                (std::sqrt(expected_second[index] / second_correction) +
                 1.0e-8F);
        }
    }
    EXPECT_EQ(parameter.data().to_vector(), expected_parameter);
    const auto state = optimizer.state();
    EXPECT_EQ(state.step, 100U);
    EXPECT_EQ(state.first_moments[0].to_vector(), expected_first);
    EXPECT_EQ(state.second_moments[0].to_vector(), expected_second);
}

TEST(AdamWTest, RestoredStateKeepsDerivedBf16MirrorFresh) {
    const AdamWConfig config{.learning_rate = 0.01F, .weight_decay = 0.0F};
    Value first_master(Tensor::from_vector({1.0F, -2.0F}, {2}), true);
    auto first_mirror = first_master.data().cast(DType::BFloat16);
    AdamW first({&first_master}, config, {{&first_master, &first_mirror}});
    sum(scale(first_master, 3.0F)).backward();
    first.step();
    first.zero_grad();

    Value restored_master(
        Tensor::from_vector(first_master.data().to_vector(), {2}), true);
    auto restored_mirror = restored_master.data().cast(DType::BFloat16);
    AdamW restored({&restored_master}, config,
                   {{&restored_master, &restored_mirror}});
    ops::fill_(restored_mirror, 99.0F);
    restored.load_state(first.state());
    EXPECT_EQ(restored_mirror.cast(DType::Float32).to_vector(),
              restored_master.data().cast(DType::BFloat16).cast(DType::Float32).to_vector());
    sum(scale(first_master, -2.0F)).backward();
    sum(scale(restored_master, -2.0F)).backward();
    first.step();
    restored.step();
    EXPECT_EQ(restored_master.data().to_vector(), first_master.data().to_vector());
    EXPECT_EQ(restored_mirror.to_vector(), first_mirror.to_vector());
}

TEST(OptimizerTest, RejectsInvalidParametersAndState) {
    Value constant(Tensor::from_vector({1}, {1}), false);
    EXPECT_THROW((void)SGD({&constant}, 0.1F), std::invalid_argument);
    Value parameter(Tensor::from_vector({1}, {1}), true);
    AdamW optimizer({&parameter});
    EXPECT_THROW(optimizer.load_state({}), std::invalid_argument);
    EXPECT_THROW((void)ops::AdamWMultiTensorWorkspace({1}, Device::cpu()),
                 std::invalid_argument);
    const ops::AdamWMultiTensorWorkspace undefined;
    EXPECT_EQ(ops::adamw_multi_tensor_workspace_stats(undefined).tensors, 0U);
    const ops::AdamWGraphStepState undefined_graph_state;
    EXPECT_FALSE(undefined_graph_state.defined());
    EXPECT_THROW((void)undefined_graph_state.synchronized_step(),
                 std::logic_error);
    EXPECT_THROW((void)optimizer.make_graph_step_state(),
                 std::invalid_argument);
    EXPECT_THROW(optimizer.synchronize_graph_step(undefined_graph_state),
                 std::logic_error);
}

}  // namespace microllm::training
