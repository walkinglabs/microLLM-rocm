#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <microllm/autograd/autograd.h>
#include <microllm/ops/ops.h>

namespace microllm::training {

class AdamW;

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

class AdamWGraphWorkspace {
public:
    AdamWGraphWorkspace() = default;
    AdamWGraphWorkspace(const AdamWGraphWorkspace&) = delete;
    AdamWGraphWorkspace& operator=(const AdamWGraphWorkspace&) = delete;
    AdamWGraphWorkspace(AdamWGraphWorkspace&&) noexcept = default;
    AdamWGraphWorkspace& operator=(AdamWGraphWorkspace&&) noexcept = default;

private:
    ops::AdamWGraphStepState step_state_;
    ops::AdamWMultiTensorWorkspace multi_tensor_;
    const AdamW* owner_ = nullptr;
    friend class AdamW;
};

class AdamW {
public:
    AdamW(Parameters parameters, AdamWConfig config = {},
          Bf16ParameterMirrors bf16_mirrors = {},
          ops::AdamWImplementation implementation = ops::AdamWImplementation::Auto,
          std::int64_t bf16_multi_tensor_threshold = -1);
    AdamW(const AdamW&) = delete;
    AdamW& operator=(const AdamW&) = delete;
    AdamW(AdamW&&) noexcept = default;
    AdamW& operator=(AdamW&&) noexcept = default;
    void step();
    [[nodiscard]] ops::AdamWGraphStepState make_graph_step_state() const;
    [[nodiscard]] AdamWGraphWorkspace make_graph_workspace();
    void step_graph_replayable(ops::AdamWGraphStepState& graph_state,
                               ops::OpContext context = {});
    void step_graph_replayable(AdamWGraphWorkspace& workspace,
                               ops::OpContext context = {});
    void synchronize_graph_step(const ops::AdamWGraphStepState& graph_state);
    void synchronize_graph_step(const AdamWGraphWorkspace& workspace);
    void zero_grad();

    [[nodiscard]] const AdamWConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::uint64_t step_count() const noexcept { return state_.step; }
    [[nodiscard]] std::uint64_t moment_state_bytes() const;
    [[nodiscard]] std::int64_t bf16_multi_tensor_threshold() const noexcept {
        return bf16_multi_tensor_threshold_;
    }
    [[nodiscard]] std::size_t bf16_multi_tensor_count() const noexcept {
        return bf16_multi_tensor_indices_.size();
    }
    [[nodiscard]] std::uint64_t bf16_multi_tensor_elements() const noexcept {
        return bf16_multi_tensor_elements_;
    }
    [[nodiscard]] AdamWState state() const;
    void load_state(AdamWState state);

private:
    Parameters parameters_;
    AdamWConfig config_;
    AdamWState state_;
    std::vector<Tensor*> bf16_mirrors_;
    ops::AdamWImplementation implementation_;
    std::int64_t bf16_multi_tensor_threshold_ = 0;
    std::unique_ptr<ops::AdamWMultiTensorWorkspace> bf16_multi_tensor_workspace_;
    std::vector<std::size_t> bf16_multi_tensor_indices_;
    std::uint64_t bf16_multi_tensor_elements_ = 0;
    bool graph_step_pending_ = false;
    const ops::AdamWGraphStepState* pending_graph_step_state_ = nullptr;
};

}  // namespace microllm::training
