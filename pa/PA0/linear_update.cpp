#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

struct Sample {
    float x;
    float y;
};

struct Result {
    float initial_loss = 0.0F;
    float final_loss = 0.0F;
    float weight = 0.0F;
    float bias = 0.0F;
    std::vector<float> initial_sample_losses;
    std::vector<float> final_sample_losses;
};

Result update_once(const std::vector<Sample>& samples, float learning_rate) {
    float weight = 0.0F;
    float bias = 0.0F;
    float weight_gradient = 0.0F;
    float bias_gradient = 0.0F;
    Result result;
    for (const auto& sample : samples) {
        const auto residual = weight * sample.x + bias - sample.y;
        result.initial_sample_losses.push_back(residual * residual);
        weight_gradient += 2.0F * residual * sample.x / static_cast<float>(samples.size());
        bias_gradient += 2.0F * residual / static_cast<float>(samples.size());
    }
    weight -= learning_rate * weight_gradient;
    bias -= learning_rate * bias_gradient;
    for (const auto& sample : samples) {
        const auto residual = weight * sample.x + bias - sample.y;
        result.final_sample_losses.push_back(residual * residual);
    }
    for (const auto value : result.initial_sample_losses) result.initial_loss += value;
    for (const auto value : result.final_sample_losses) result.final_loss += value;
    result.initial_loss /= static_cast<float>(samples.size());
    result.final_loss /= static_cast<float>(samples.size());
    result.weight = weight;
    result.bias = bias;
    return result;
}

}  // namespace

int main() {
    try {
        const auto regular = update_once({{1, 2}, {2, 4}, {3, 6}}, 0.1F);
        const auto with_outlier = update_once({{1, 2}, {2, 4}, {3, 6}, {3, -10}}, 0.1F);
        std::cout << "regular_initial_loss=" << regular.initial_loss << '\n';
        std::cout << "regular_final_loss=" << regular.final_loss << '\n';
        std::cout << "regular_weight=" << regular.weight << " regular_bias=" << regular.bias << '\n';
        std::cout << "outlier_initial_loss=" << with_outlier.initial_loss << '\n';
        std::cout << "outlier_final_loss=" << with_outlier.final_loss << '\n';
        std::cout << "outlier_weight=" << with_outlier.weight
                  << " outlier_bias=" << with_outlier.bias << '\n';
        bool regular_sample_worsened = false;
        for (std::size_t index = 0; index < 3; ++index) {
            const auto worsened = with_outlier.final_sample_losses[index] >
                                  with_outlier.initial_sample_losses[index];
            regular_sample_worsened = regular_sample_worsened || worsened;
            std::cout << "sample=" << index << " initial="
                      << with_outlier.initial_sample_losses[index] << " final="
                      << with_outlier.final_sample_losses[index]
                      << " worsened=" << (worsened ? "true" : "false") << '\n';
        }
        if (!(regular.final_loss < regular.initial_loss) ||
            !(with_outlier.final_loss < with_outlier.initial_loss) ||
            !regular_sample_worsened || !std::isfinite(with_outlier.final_loss)) {
            throw std::runtime_error("PA0 acceptance conditions failed");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "pa0_linear: " << error.what() << '\n';
        return 1;
    }
}
