#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <microllm/autograd/autograd.h>
#include <microllm/ops/ops.h>

namespace microllm::training {

using Parameters = std::vector<autograd::Value*>;
using Bf16ParameterMirrors =
    std::vector<std::pair<autograd::Value*, Tensor*>>;

void zero_grad(const Parameters& parameters);

class SGD {
public:
    SGD(Parameters parameters, float learning_rate, float weight_decay = 0.0F);
    void step();
    void zero_grad();

private:
    Parameters parameters_;
    float learning_rate_;
    float weight_decay_;
};

struct AdamWConfig {
    float learning_rate = 1.0e-3F;
    float beta1 = 0.9F;
    float beta2 = 0.999F;
    float epsilon = 1.0e-8F;
    float weight_decay = 0.01F;
    enum class MomentPrecision : std::uint8_t {
        Float32 = 0,
        BFloat16 = 1,
    };
    MomentPrecision moment_precision = MomentPrecision::Float32;
};

struct AdamWState {
    std::uint64_t step = 0;
    std::vector<Tensor> first_moments;
    std::vector<Tensor> second_moments;
};

class AdamW {
public:
    AdamW(Parameters parameters, AdamWConfig config = {},
          Bf16ParameterMirrors bf16_mirrors = {},
          ops::AdamWImplementation implementation = ops::AdamWImplementation::Auto);
    void step();
    void zero_grad();

    [[nodiscard]] const AdamWConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::uint64_t moment_state_bytes() const;
    [[nodiscard]] AdamWState state() const;
    void load_state(AdamWState state);

private:
    Parameters parameters_;
    AdamWConfig config_;
    AdamWState state_;
    std::vector<Tensor*> bf16_mirrors_;
    ops::AdamWImplementation implementation_;
};

}  // namespace microllm::training
