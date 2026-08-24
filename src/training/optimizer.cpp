#include <microllm/training/optimizer.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <microllm/ops/ops.h>

namespace microllm::training {
namespace {

void validate_parameters(const Parameters& parameters) {
    for (const auto* parameter : parameters) {
        if (parameter == nullptr || !parameter->defined()) {
            throw std::invalid_argument("optimizer parameters must be non-null defined Values");
        }
        if (!parameter->requires_grad()) {
            throw std::invalid_argument("optimizer parameters must require gradients");
        }
        if (parameter->data().dtype() != DType::Float32 || !parameter->data().is_contiguous()) {
            throw std::invalid_argument("optimizer parameters must be contiguous float32");
        }
    }
}

void validate_hyperparameter(float value, float lower, float upper, const char* name,
                             bool upper_inclusive = false) {
    const auto valid_upper = upper_inclusive ? value <= upper : value < upper;
    if (!(value >= lower && valid_upper)) {
        throw std::invalid_argument(std::string(name) + " is outside its valid range");
    }
}

}  // namespace

void zero_grad(const Parameters& parameters) {
    for (auto* parameter : parameters) {
        if (parameter != nullptr) parameter->zero_grad();
    }
}

SGD::SGD(Parameters parameters, float learning_rate, float weight_decay)
    : parameters_(std::move(parameters)),
      learning_rate_(learning_rate),
      weight_decay_(weight_decay) {
    validate_parameters(parameters_);
    if (!(learning_rate_ > 0.0F)) throw std::invalid_argument("SGD learning rate must be positive");
    if (weight_decay_ < 0.0F) throw std::invalid_argument("SGD weight decay must be non-negative");
}

void SGD::step() {
    for (auto* parameter : parameters_) {
        if (!parameter->has_grad()) continue;
        if (parameter->grad().shape() != parameter->data().shape()) {
            throw std::invalid_argument("SGD gradient shape mismatch");
        }
        auto values = parameter->data().to_vector();
        const auto gradients = parameter->grad().to_vector();
        for (std::int64_t index = 0; index < parameter->data().numel(); ++index) {
            const auto offset = static_cast<std::size_t>(index);
            values[offset] -= learning_rate_ *
                              (gradients[offset] + weight_decay_ * values[offset]);
        }
        const auto device = parameter->data().device();
        parameter->mutable_data() = Tensor::from_vector(values, parameter->data().shape()).to(device);
    }
}

void SGD::zero_grad() { training::zero_grad(parameters_); }

AdamW::AdamW(Parameters parameters, AdamWConfig config,
             Bf16ParameterMirrors bf16_mirrors,
             ops::AdamWImplementation implementation,
             std::int64_t bf16_multi_tensor_threshold)
    : parameters_(std::move(parameters)), config_(config),
      bf16_mirrors_(parameters_.size(), nullptr), implementation_(implementation),
      bf16_multi_tensor_threshold_(bf16_multi_tensor_threshold) {
    validate_parameters(parameters_);
    if (!(config_.learning_rate > 0.0F)) {
        throw std::invalid_argument("AdamW learning rate must be positive");
    }
    validate_hyperparameter(config_.beta1, 0.0F, 1.0F, "AdamW beta1");
    validate_hyperparameter(config_.beta2, 0.0F, 1.0F, "AdamW beta2");
    if (!(config_.epsilon > 0.0F)) throw std::invalid_argument("AdamW epsilon must be positive");
    if (config_.weight_decay < 0.0F) {
        throw std::invalid_argument("AdamW weight decay must be non-negative");
    }
    if (config_.moment_precision == AdamWConfig::MomentPrecision::BFloat16 &&
        implementation_ == ops::AdamWImplementation::Vectorized) {
        throw std::invalid_argument(
            "BF16 AdamW moments do not support the FP32 vectorized implementation selector");
    }
    if (bf16_multi_tensor_threshold_ < -1) {
        throw std::invalid_argument(
            "BF16 AdamW multi-tensor threshold must be auto (-1) or non-negative");
    }
    if (bf16_multi_tensor_threshold_ == -1) {
        bf16_multi_tensor_threshold_ =
            config_.moment_precision == AdamWConfig::MomentPrecision::BFloat16 &&
                    !parameters_.empty() &&
                    parameters_.front()->data().device().is_hip()
                ? 1 << 20
                : 0;
    }
    if (bf16_multi_tensor_threshold_ > 0 &&
        config_.moment_precision != AdamWConfig::MomentPrecision::BFloat16) {
        throw std::invalid_argument(
            "AdamW multi-tensor threshold requires BF16 moments");
    }
    const auto moment_dtype =
        config_.moment_precision == AdamWConfig::MomentPrecision::BFloat16
            ? DType::BFloat16
            : DType::Float32;
    state_.first_moments.reserve(parameters_.size());
    state_.second_moments.reserve(parameters_.size());
    for (const auto* parameter : parameters_) {
        Tensor first(parameter->data().shape(), moment_dtype, parameter->data().device());
        Tensor second(parameter->data().shape(), moment_dtype, parameter->data().device());
        ops::fill_(first, 0.0F);
        ops::fill_(second, 0.0F);
        state_.first_moments.push_back(std::move(first));
        state_.second_moments.push_back(std::move(second));
    }
    std::unordered_map<autograd::Value*, std::size_t> indices;
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        indices.emplace(parameters_[index], index);
    }
    for (const auto& [master, mirror] : bf16_mirrors) {
        const auto found = indices.find(master);
        if (found == indices.end() || mirror == nullptr || !mirror->defined() ||
            mirror->dtype() != DType::BFloat16 ||
            mirror->shape() != master->data().shape() ||
            mirror->device() != master->data().device() || !mirror->is_contiguous() ||
            bf16_mirrors_[found->second] != nullptr) {
            throw std::invalid_argument("AdamW BF16 mirror mapping is invalid");
        }
        bf16_mirrors_[found->second] = mirror;
    }
    if (bf16_multi_tensor_threshold_ > 0) {
        if (parameters_.empty() || !parameters_.front()->data().device().is_hip()) {
            throw std::invalid_argument(
                "BF16 AdamW multi-tensor threshold requires HIP parameters");
        }
        const auto device = parameters_.front()->data().device();
        std::vector<std::int64_t> element_counts;
        for (std::size_t index = 0; index < parameters_.size(); ++index) {
            const auto& data = parameters_[index]->data();
            if (data.device() != device) {
                throw std::invalid_argument(
                    "BF16 AdamW multi-tensor parameters must share one device");
            }
            if (data.numel() <= bf16_multi_tensor_threshold_) {
                bf16_multi_tensor_indices_.push_back(index);
                element_counts.push_back(data.numel());
                bf16_multi_tensor_elements_ +=
                    static_cast<std::uint64_t>(data.numel());
            }
        }
        if (!element_counts.empty()) {
            bf16_multi_tensor_workspace_ =
                std::make_unique<ops::AdamWMultiTensorWorkspace>(
                    std::move(element_counts), device);
        }
    }
}

void AdamW::step() {
    if (graph_step_pending_) {
        throw std::logic_error(
            "synchronize the AdamW Graph step before ordinary step()");
    }
    ++state_.step;
    const auto first_correction = 1.0F - std::pow(config_.beta1, static_cast<float>(state_.step));
    const auto second_correction = 1.0F - std::pow(config_.beta2, static_cast<float>(state_.step));
    const ops::OpContext training_context{.mode = ops::OpMode::Training};
    for (std::size_t parameter_index = 0; parameter_index < parameters_.size(); ++parameter_index) {
        auto* parameter = parameters_[parameter_index];
        if (bf16_multi_tensor_workspace_ != nullptr &&
            parameter->data().numel() <= bf16_multi_tensor_threshold_) {
            continue;
        }
        if (!parameter->has_grad()) continue;
        if (parameter->grad().shape() != parameter->data().shape()) {
            throw std::invalid_argument("AdamW gradient shape mismatch");
        }
        if (config_.moment_precision ==
            AdamWConfig::MomentPrecision::BFloat16) {
            ops::adamw_update_bf16_moments_(
                parameter->mutable_data(), parameter->grad(),
                state_.first_moments[parameter_index],
                state_.second_moments[parameter_index],
                bf16_mirrors_[parameter_index], config_.learning_rate,
                config_.beta1, config_.beta2, config_.epsilon,
                config_.weight_decay, first_correction, second_correction,
                training_context);
        } else if (bf16_mirrors_[parameter_index] != nullptr) {
            ops::adamw_update_bf16_mirror_(
                parameter->mutable_data(), parameter->grad(),
                state_.first_moments[parameter_index],
                state_.second_moments[parameter_index],
                *bf16_mirrors_[parameter_index], config_.learning_rate,
                config_.beta1, config_.beta2, config_.epsilon,
                config_.weight_decay, first_correction, second_correction,
                training_context, implementation_);
        } else {
            ops::adamw_update_(parameter->mutable_data(), parameter->grad(),
                               state_.first_moments[parameter_index],
                               state_.second_moments[parameter_index],
                               config_.learning_rate, config_.beta1, config_.beta2,
                               config_.epsilon, config_.weight_decay, first_correction,
                               second_correction, training_context, implementation_);
        }
    }
    if (bf16_multi_tensor_workspace_ != nullptr) {
        std::vector<ops::AdamWMultiTensorEntry> entries;
        entries.reserve(bf16_multi_tensor_indices_.size());
        bool any_gradient = false;
        for (const auto parameter_index : bf16_multi_tensor_indices_) {
            auto* parameter = parameters_[parameter_index];
            if (parameter->has_grad() &&
                parameter->grad().shape() != parameter->data().shape()) {
                throw std::invalid_argument("AdamW gradient shape mismatch");
            }
            any_gradient = any_gradient || parameter->has_grad();
            entries.push_back({
                .parameter = &parameter->mutable_data(),
                .gradient = parameter->has_grad() ? &parameter->grad() : nullptr,
                .first_moment = &state_.first_moments[parameter_index],
                .second_moment = &state_.second_moments[parameter_index],
                .bf16_mirror = bf16_mirrors_[parameter_index],
            });
        }
        if (any_gradient) {
            ops::adamw_update_multi_(
                *bf16_multi_tensor_workspace_, entries,
                config_.learning_rate, config_.beta1, config_.beta2,
                config_.epsilon, config_.weight_decay, first_correction,
                second_correction, training_context);
        }
    }
}

ops::AdamWGraphStepState AdamW::make_graph_step_state() const {
    if (graph_step_pending_) {
        throw std::logic_error(
            "cannot replace an unsynchronized AdamW Graph step state");
    }
    if (parameters_.empty() || !parameters_.front()->data().device().is_hip()) {
        throw std::invalid_argument(
            "AdamW Graph replay requires HIP parameters");
    }
    const auto device = parameters_.front()->data().device();
    for (const auto* parameter : parameters_) {
        if (parameter->data().device() != device) {
            throw std::invalid_argument(
                "AdamW Graph replay parameters must share one device");
        }
    }
    return ops::AdamWGraphStepState(device, state_.step);
}

AdamWGraphWorkspace AdamW::make_graph_workspace() {
    AdamWGraphWorkspace result;
    result.owner_ = this;
    result.step_state_ = make_graph_step_state();
    std::vector<std::int64_t> element_counts;
    element_counts.reserve(parameters_.size());
    for (const auto* parameter : parameters_) {
        element_counts.push_back(parameter->data().numel());
    }
    result.multi_tensor_ = ops::AdamWMultiTensorWorkspace(
        std::move(element_counts), result.step_state_.device());
    std::vector<ops::AdamWMultiTensorEntry> entries;
    entries.reserve(parameters_.size());
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        auto* parameter = parameters_[index];
        entries.push_back({
            .parameter = &parameter->mutable_data(),
            .gradient = parameter->has_grad() ? &parameter->grad() : nullptr,
            .first_moment = &state_.first_moments[index],
            .second_moment = &state_.second_moments[index],
            .bf16_mirror = bf16_mirrors_[index],
        });
    }
    ops::adamw_prepare_multi_graph_(result.multi_tensor_, entries);
    return result;
}

void AdamW::step_graph_replayable(ops::AdamWGraphStepState& graph_state,
                                  ops::OpContext context) {
    if (graph_step_pending_) {
        throw std::logic_error(
            "synchronize the existing AdamW Graph step before recapture");
    }
    if (parameters_.empty() || !parameters_.front()->data().device().is_hip() ||
        !graph_state.defined() ||
        graph_state.device() != parameters_.front()->data().device()) {
        throw std::invalid_argument(
            "AdamW Graph replay state must match HIP parameters");
    }
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        const auto* parameter = parameters_[index];
        if (parameter->data().device() != graph_state.device()) {
            throw std::invalid_argument(
                "AdamW Graph replay parameters must share one device");
        }
        if (parameter->has_grad() &&
            parameter->grad().shape() != parameter->data().shape()) {
            throw std::invalid_argument("AdamW gradient shape mismatch");
        }
    }
    context.mode = ops::OpMode::Training;
    // Mark host state stale before the first enqueue. If a later capture or
    // launch validation fails, callers must still synchronize/inspect the
    // device step before returning to checkpoint or ordinary step semantics.
    graph_step_pending_ = true;
    pending_graph_step_state_ = &graph_state;
    ops::adamw_graph_advance_(graph_state, config_.beta1, config_.beta2,
                              context);
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        auto* parameter = parameters_[index];
        if (!parameter->has_grad()) continue;
        if (config_.moment_precision ==
            AdamWConfig::MomentPrecision::BFloat16) {
            ops::adamw_update_bf16_moments_graph_(
                parameter->mutable_data(), parameter->grad(),
                state_.first_moments[index], state_.second_moments[index],
                bf16_mirrors_[index], graph_state, config_.learning_rate,
                config_.beta1, config_.beta2, config_.epsilon,
                config_.weight_decay, context);
        } else {
            ops::adamw_update_graph_(
                parameter->mutable_data(), parameter->grad(),
                state_.first_moments[index], state_.second_moments[index],
                bf16_mirrors_[index], graph_state, config_.learning_rate,
                config_.beta1, config_.beta2, config_.epsilon,
                config_.weight_decay, context, implementation_);
        }
    }
}

void AdamW::step_graph_replayable(AdamWGraphWorkspace& workspace,
                                  ops::OpContext context) {
    if (graph_step_pending_) {
        throw std::logic_error(
            "synchronize the existing AdamW Graph step before recapture");
    }
    const auto stats =
        ops::adamw_multi_tensor_workspace_stats(workspace.multi_tensor_);
    if (!workspace.step_state_.defined() ||
        workspace.owner_ != this ||
        stats.tensors != parameters_.size() ||
        !stats.graph_descriptors_prepared || parameters_.empty() ||
        workspace.step_state_.device() != parameters_.front()->data().device()) {
        throw std::invalid_argument(
            "AdamW multi-tensor Graph workspace does not match optimizer");
    }
    context.mode = ops::OpMode::Training;
    graph_step_pending_ = true;
    pending_graph_step_state_ = &workspace.step_state_;
    ops::adamw_graph_advance_(workspace.step_state_, config_.beta1,
                              config_.beta2, context);
    ops::adamw_update_multi_graph_(
        workspace.multi_tensor_, workspace.step_state_, config_.learning_rate,
        config_.beta1, config_.beta2, config_.epsilon,
        config_.weight_decay, context);
}

void AdamW::synchronize_graph_step(
    const ops::AdamWGraphStepState& graph_state) {
    if (!graph_step_pending_) {
        throw std::logic_error("AdamW has no pending Graph step to synchronize");
    }
    if (!graph_state.defined() || parameters_.empty() ||
        graph_state.device() != parameters_.front()->data().device() ||
        pending_graph_step_state_ != &graph_state) {
        throw std::invalid_argument("AdamW Graph step state does not match optimizer");
    }
    state_.step = graph_state.synchronized_step();
    graph_step_pending_ = false;
    pending_graph_step_state_ = nullptr;
}

void AdamW::synchronize_graph_step(
    const AdamWGraphWorkspace& workspace) {
    synchronize_graph_step(workspace.step_state_);
}

void AdamW::zero_grad() { training::zero_grad(parameters_); }

std::uint64_t AdamW::moment_state_bytes() const {
    std::uint64_t bytes = 0;
    for (const auto& moment : state_.first_moments) {
        bytes += static_cast<std::uint64_t>(moment.storage().num_bytes());
    }
    for (const auto& moment : state_.second_moments) {
        bytes += static_cast<std::uint64_t>(moment.storage().num_bytes());
    }
    return bytes;
}

AdamWState AdamW::state() const {
    if (graph_step_pending_) {
        throw std::logic_error(
            "synchronize the AdamW Graph step before reading state");
    }
    AdamWState snapshot;
    snapshot.step = state_.step;
    snapshot.first_moments.reserve(state_.first_moments.size());
    snapshot.second_moments.reserve(state_.second_moments.size());
    for (const auto& moment : state_.first_moments) {
        snapshot.first_moments.push_back(Tensor::from_vector(moment.to_vector(), moment.shape()));
    }
    for (const auto& moment : state_.second_moments) {
        snapshot.second_moments.push_back(Tensor::from_vector(moment.to_vector(), moment.shape()));
    }
    return snapshot;
}

void AdamW::load_state(AdamWState state) {
    if (graph_step_pending_) {
        throw std::logic_error(
            "synchronize the AdamW Graph step before loading state");
    }
    if (state.first_moments.size() != parameters_.size() ||
        state.second_moments.size() != parameters_.size()) {
        throw std::invalid_argument("AdamW state parameter count mismatch");
    }
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        if (state.first_moments[index].shape() != parameters_[index]->data().shape() ||
            state.second_moments[index].shape() != parameters_[index]->data().shape()) {
            throw std::invalid_argument("AdamW state shape mismatch");
        }
        if (state.first_moments[index].dtype() != DType::Float32 ||
            state.second_moments[index].dtype() != DType::Float32) {
            throw std::invalid_argument("AdamW state must be float32");
        }
        const auto device = parameters_[index]->data().device();
        const auto dtype =
            config_.moment_precision == AdamWConfig::MomentPrecision::BFloat16
                ? DType::BFloat16
                : DType::Float32;
        state.first_moments[index] = state.first_moments[index].to(device).cast(dtype);
        state.second_moments[index] = state.second_moments[index].to(device).cast(dtype);
    }
    state_ = std::move(state);
    // BF16 forward copies are derived state: checkpoints keep the FP32 master
    // weights only.  restore_checkpoint() loads those masters before it calls
    // this method, so refresh every registered mirror here rather than storing
    // a second, potentially stale, copy in the checkpoint.
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        if (bf16_mirrors_[index] != nullptr) {
            *bf16_mirrors_[index] = parameters_[index]->data().cast(DType::BFloat16);
        }
    }
}

}  // namespace microllm::training
