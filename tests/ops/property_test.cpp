#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/autograd/autograd.h>
#include <microllm/ops/ops.h>

namespace microllm::ops {
namespace {

std::int64_t shape_numel(const Shape& shape) {
    return std::accumulate(shape.begin(), shape.end(), std::int64_t{1},
                           std::multiplies<>());
}

std::vector<float> random_values(std::mt19937& generator, std::int64_t count,
                                 float low = -3.0F, float high = 3.0F) {
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<float> values(static_cast<std::size_t>(count));
    for (auto& value : values) value = distribution(generator);
    return values;
}

void expect_near_values(const std::vector<float>& actual,
                        const std::vector<float>& expected,
                        float tolerance = 2.0e-5F) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "index=" << index;
    }
}

float composite_loss(const std::vector<float>& input_values,
                     const std::vector<float>& weight_values,
                     const std::vector<float>& up_values,
                     const Shape& shape) {
    const auto width = shape.back();
    return reduce_sum(swiglu(
        rms_norm(Tensor::from_vector(input_values, shape),
                 Tensor::from_vector(weight_values, {width})),
        Tensor::from_vector(up_values, shape))).to_vector()[0];
}

}  // namespace

TEST(OperatorPropertyTest, ElementwiseOpsMatchScalarReferenceAcrossRanks) {
    std::mt19937 generator(0x4d4c4c4dU);
    const std::vector<Shape> shapes{{}, {1}, {7}, {2, 3}, {2, 1, 5}, {2, 2, 3, 4}};
    for (const auto& shape : shapes) {
        SCOPED_TRACE(::testing::Message() << "rank=" << shape.size());
        const auto count = shape_numel(shape);
        const auto left_values = random_values(generator, count);
        const auto right_values = random_values(generator, count);
        const auto left = Tensor::from_vector(left_values, shape);
        const auto right = Tensor::from_vector(right_values, shape);
        std::vector<float> expected_add(left_values.size());
        std::vector<float> expected_multiply(left_values.size());
        std::vector<float> expected_scale(left_values.size());
        for (std::size_t index = 0; index < left_values.size(); ++index) {
            expected_add[index] = left_values[index] + right_values[index];
            expected_multiply[index] = left_values[index] * right_values[index];
            expected_scale[index] = left_values[index] * -0.375F;
        }
        expect_near_values(add(left, right).to_vector(), expected_add, 0.0F);
        expect_near_values(multiply(left, right).to_vector(), expected_multiply, 0.0F);
        expect_near_values(scale(left, -0.375F).to_vector(), expected_scale, 0.0F);
    }
}

TEST(OperatorPropertyTest, MatmulMatchesIndependentLoopForEdgeAndBatchShapes) {
    std::mt19937 generator(0x4d41544dU);
    const std::vector<std::pair<Shape, Shape>> cases{
        {{1, 1}, {1, 1}},
        {{1, 7}, {7, 1}},
        {{3, 5}, {5, 2}},
        {{2, 1, 3}, {2, 3, 4}},
        {{2, 2, 3, 4}, {2, 2, 4, 2}},
        {{2, 0}, {0, 3}},
    };
    for (const auto& [left_shape, right_shape] : cases) {
        SCOPED_TRACE(::testing::Message() << "rank=" << left_shape.size());
        const auto left_values = random_values(generator, shape_numel(left_shape));
        const auto right_values = random_values(generator, shape_numel(right_shape));
        const auto rank = left_shape.size();
        const auto rows = left_shape[rank - 2];
        const auto inner = left_shape[rank - 1];
        const auto columns = right_shape[rank - 1];
        std::int64_t batches = 1;
        for (std::size_t dim = 0; dim + 2 < rank; ++dim) batches *= left_shape[dim];
        std::vector<float> expected(static_cast<std::size_t>(batches * rows * columns), 0.0F);
        for (std::int64_t batch = 0; batch < batches; ++batch) {
            for (std::int64_t row = 0; row < rows; ++row) {
                for (std::int64_t column = 0; column < columns; ++column) {
                    for (std::int64_t reduction = 0; reduction < inner; ++reduction) {
                        expected[static_cast<std::size_t>(
                            batch * rows * columns + row * columns + column)] +=
                            left_values[static_cast<std::size_t>(
                                batch * rows * inner + row * inner + reduction)] *
                            right_values[static_cast<std::size_t>(
                                batch * inner * columns + reduction * columns + column)];
                    }
                }
            }
        }
        const auto actual = matmul(Tensor::from_vector(left_values, left_shape),
                                   Tensor::from_vector(right_values, right_shape));
        Shape expected_shape(left_shape.begin(), left_shape.end() - 2);
        expected_shape.insert(expected_shape.end(), {rows, columns});
        EXPECT_EQ(actual.shape(), expected_shape);
        expect_near_values(actual.to_vector(), expected, 1.0e-5F);
    }
}

TEST(OperatorPropertyTest, NonlinearOpsSatisfyShapeAndNumericProperties) {
    std::mt19937 generator(0x4e4c4f50U);
    const std::vector<Shape> shapes{{1, 1}, {2, 3}, {2, 2, 5}, {1, 3, 2, 8}};
    for (const auto& shape : shapes) {
        SCOPED_TRACE(::testing::Message() << "rank=" << shape.size()
                                         << " width=" << shape.back());
        auto values = random_values(generator, shape_numel(shape), -8.0F, 8.0F);
        if (values.size() >= 2) {
            values[0] = 80.0F;
            values[1] = -80.0F;
        }
        const auto width = shape.back();
        const auto rows = shape_numel(shape) / width;
        const auto input = Tensor::from_vector(values, shape);
        const auto probabilities = softmax(input).to_vector();
        for (std::int64_t row = 0; row < rows; ++row) {
            float sum = 0.0F;
            for (std::int64_t column = 0; column < width; ++column) {
                const auto value = probabilities[static_cast<std::size_t>(row * width + column)];
                EXPECT_TRUE(std::isfinite(value));
                EXPECT_GE(value, 0.0F);
                sum += value;
            }
            EXPECT_NEAR(sum, 1.0F, 2.0e-6F);
        }

        const auto weights = random_values(generator, width, 0.25F, 2.0F);
        const auto normalized = rms_norm(input, Tensor::from_vector(weights, {width})).to_vector();
        for (std::int64_t row = 0; row < rows; ++row) {
            float square_sum = 0.0F;
            for (std::int64_t column = 0; column < width; ++column) {
                const auto value = values[static_cast<std::size_t>(row * width + column)];
                square_sum += value * value;
            }
            const auto divisor = std::sqrt(square_sum / static_cast<float>(width) + 1.0e-5F);
            for (std::int64_t column = 0; column < width; ++column) {
                const auto index = static_cast<std::size_t>(row * width + column);
                EXPECT_NEAR(normalized[index], values[index] / divisor * weights[column], 2.0e-5F);
            }
        }

        const auto up_values = random_values(generator, shape_numel(shape));
        const auto activated = silu(input).to_vector();
        const auto gated = swiglu(input, Tensor::from_vector(up_values, shape)).to_vector();
        for (std::size_t index = 0; index < values.size(); ++index) {
            const auto expected = values[index] / (1.0F + std::exp(-values[index]));
            EXPECT_NEAR(activated[index], expected, 2.0e-5F);
            EXPECT_NEAR(gated[index], expected * up_values[index], 3.0e-5F);
        }
    }
}

TEST(OperatorPropertyTest, IndexSequenceAndLossOpsCoverVariedShapes) {
    std::mt19937 generator(0x494e4458U);
    const auto weight_values = random_values(generator, 11 * 7);
    const auto weight = Tensor::from_vector(weight_values, {11, 7});
    for (const Shape& index_shape : std::vector<Shape>{{}, {1}, {2, 3}}) {
        std::vector<std::int32_t> indices(static_cast<std::size_t>(shape_numel(index_shape)));
        for (std::size_t index = 0; index < indices.size(); ++index) {
            indices[index] = static_cast<std::int32_t>((index * 7 + 3) % 11);
        }
        const auto output = embedding(
            weight, Tensor::from_int32_vector(indices, index_shape));
        auto expected_shape = index_shape;
        expected_shape.push_back(7);
        EXPECT_EQ(output.shape(), expected_shape);
        const auto output_values = output.to_vector();
        for (std::size_t row = 0; row < indices.size(); ++row) {
            for (std::size_t column = 0; column < 7; ++column) {
                EXPECT_FLOAT_EQ(output_values[row * 7 + column],
                                weight_values[static_cast<std::size_t>(indices[row]) * 7 + column]);
            }
        }
    }

    for (const auto sequence : {1, 2, 5}) {
        const Shape score_shape{2, 3, sequence, sequence};
        const auto scores = Tensor::from_vector(
            random_values(generator, shape_numel(score_shape)), score_shape);
        const auto probabilities = causal_softmax(scores).to_vector();
        for (std::int64_t row = 0; row < 6 * sequence; ++row) {
            const auto query = row % sequence;
            float sum = 0.0F;
            for (std::int64_t key = 0; key < sequence; ++key) {
                const auto probability = probabilities[static_cast<std::size_t>(row * sequence + key)];
                if (key > query) {
                    EXPECT_FLOAT_EQ(probability, 0.0F);
                }
                sum += probability;
            }
            EXPECT_NEAR(sum, 1.0F, 2.0e-6F);
        }
    }

    for (const Shape& logit_shape : std::vector<Shape>{{1, 2}, {2, 5}, {2, 3, 7}}) {
        const auto vocabulary = logit_shape.back();
        const auto rows = shape_numel(logit_shape) / vocabulary;
        const auto logits = random_values(generator, shape_numel(logit_shape), -20.0F, 20.0F);
        std::vector<std::int32_t> targets(static_cast<std::size_t>(rows));
        for (std::int64_t row = 0; row < rows; ++row) {
            targets[static_cast<std::size_t>(row)] =
                static_cast<std::int32_t>(row % vocabulary);
        }
        Shape target_shape(logit_shape.begin(), logit_shape.end() - 1);
        const auto loss = cross_entropy(Tensor::from_vector(logits, logit_shape),
                                        Tensor::from_int32_vector(targets, target_shape));
        EXPECT_EQ(loss.shape(), Shape{});
        EXPECT_TRUE(std::isfinite(loss.to_vector()[0]));
    }

    for (const Shape& rope_shape : std::vector<Shape>{{1, 1, 2}, {2, 3, 4}, {1, 2, 3, 8}}) {
        const auto input_values = random_values(generator, shape_numel(rope_shape));
        const auto output_values = rope(Tensor::from_vector(input_values, rope_shape)).to_vector();
        for (std::size_t index = 0; index < input_values.size(); index += 2) {
            const auto input_norm = input_values[index] * input_values[index] +
                                    input_values[index + 1] * input_values[index + 1];
            const auto output_norm = output_values[index] * output_values[index] +
                                     output_values[index + 1] * output_values[index + 1];
            EXPECT_NEAR(output_norm, input_norm, 2.0e-5F);
        }
    }
}

TEST(OperatorPropertyTest, RandomCompositeGradientsMatchFiniteDifference) {
    std::mt19937 generator(0x47524144U);
    constexpr float step = 1.0e-3F;
    for (const std::int64_t width : {1, 2, 5}) {
        const Shape shape{2, width};
        auto input_values = random_values(generator, shape_numel(shape), -1.5F, 1.5F);
        auto weight_values = random_values(generator, width, 0.5F, 1.5F);
        auto up_values = random_values(generator, shape_numel(shape), -1.5F, 1.5F);

        autograd::Value input(Tensor::from_vector(input_values, shape), true);
        autograd::Value weight(Tensor::from_vector(weight_values, {width}), true);
        autograd::Value up(Tensor::from_vector(up_values, shape), true);
        autograd::sum(autograd::swiglu(autograd::rms_norm(input, weight), up)).backward();

        const auto check = [&](std::vector<float> values,
                               const std::vector<float>& analytic,
                               auto evaluate) {
            for (std::size_t index = 0; index < values.size(); ++index) {
                const auto original = values[index];
                values[index] = original + step;
                const auto positive = evaluate(values);
                values[index] = original - step;
                const auto negative = evaluate(values);
                values[index] = original;
                const auto numerical = (positive - negative) / (2.0F * step);
                EXPECT_NEAR(analytic[index], numerical, 3.0e-3F)
                    << "width=" << width << " index=" << index;
            }
        };
        check(input_values, input.grad().to_vector(), [&](const auto& changed) {
            return composite_loss(changed, weight_values, up_values, shape);
        });
        check(weight_values, weight.grad().to_vector(), [&](const auto& changed) {
            return composite_loss(input_values, changed, up_values, shape);
        });
        check(up_values, up.grad().to_vector(), [&](const auto& changed) {
            return composite_loss(input_values, weight_values, changed, shape);
        });
    }
}

}  // namespace microllm::ops
