#include <microllm/training/optimizer.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

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
        if (!parameter->data().device().is_cpu() || parameter->data().dtype() != DType::Float32 ||
            !parameter->data().is_contiguous()) {
            throw std::invalid_argument("the first optimizer path requires contiguous CPU float32");
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
        auto* values = parameter->mutable_data().data_float();
        const auto gradients = parameter->grad().to_vector();
        for (std::int64_t index = 0; index < parameter->data().numel(); ++index) {
            const auto offset = static_cast<std::size_t>(index);
            values[offset] -= learning_rate_ *
                              (gradients[offset] + weight_decay_ * values[offset]);
        }
    }
}

void SGD::zero_grad() { training::zero_grad(parameters_); }

AdamW::AdamW(Parameters parameters, AdamWConfig config)
    : parameters_(std::move(parameters)), config_(config) {
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
    state_.first_moments.reserve(parameters_.size());
    state_.second_moments.reserve(parameters_.size());
    for (const auto* parameter : parameters_) {
        Tensor first(parameter->data().shape());
        Tensor second(parameter->data().shape());
        first.fill(0.0F);
        second.fill(0.0F);
        state_.first_moments.push_back(std::move(first));
        state_.second_moments.push_back(std::move(second));
    }
}

void AdamW::step() {
    ++state_.step;
    const auto first_correction = 1.0F - std::pow(config_.beta1, static_cast<float>(state_.step));
    const auto second_correction = 1.0F - std::pow(config_.beta2, static_cast<float>(state_.step));
    for (std::size_t parameter_index = 0; parameter_index < parameters_.size(); ++parameter_index) {
        auto* parameter = parameters_[parameter_index];
        if (!parameter->has_grad()) continue;
        if (parameter->grad().shape() != parameter->data().shape()) {
            throw std::invalid_argument("AdamW gradient shape mismatch");
        }
        auto* values = parameter->mutable_data().data_float();
        auto* first = state_.first_moments[parameter_index].data_float();
        auto* second = state_.second_moments[parameter_index].data_float();
        const auto gradients = parameter->grad().to_vector();
        for (std::int64_t index = 0; index < parameter->data().numel(); ++index) {
            const auto offset = static_cast<std::size_t>(index);
            const auto gradient = gradients[offset];
            first[offset] = config_.beta1 * first[offset] + (1.0F - config_.beta1) * gradient;
            second[offset] = config_.beta2 * second[offset] +
                             (1.0F - config_.beta2) * gradient * gradient;
            const auto corrected_first = first[offset] / first_correction;
            const auto corrected_second = second[offset] / second_correction;
            values[offset] *= 1.0F - config_.learning_rate * config_.weight_decay;
            values[offset] -= config_.learning_rate * corrected_first /
                              (std::sqrt(corrected_second) + config_.epsilon);
        }
    }
}

void AdamW::zero_grad() { training::zero_grad(parameters_); }
AdamWState AdamW::state() const {
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
    if (state.first_moments.size() != parameters_.size() ||
        state.second_moments.size() != parameters_.size()) {
        throw std::invalid_argument("AdamW state parameter count mismatch");
    }
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        if (state.first_moments[index].shape() != parameters_[index]->data().shape() ||
            state.second_moments[index].shape() != parameters_[index]->data().shape()) {
            throw std::invalid_argument("AdamW state shape mismatch");
        }
        if (!state.first_moments[index].device().is_cpu() ||
            !state.second_moments[index].device().is_cpu() ||
            state.first_moments[index].dtype() != DType::Float32 ||
            state.second_moments[index].dtype() != DType::Float32) {
            throw std::invalid_argument("AdamW state must be CPU float32");
        }
    }
    state_ = std::move(state);
}

}  // namespace microllm::training
