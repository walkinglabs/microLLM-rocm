#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <microllm/io/safetensors.h>

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument(
                "usage: microllm_compare_safetensors BASELINE CANDIDATE");
        }
        const auto baseline = microllm::io::load_safetensors(argv[1]);
        const auto candidate = microllm::io::load_safetensors(argv[2]);
        if (baseline.size() != candidate.size()) {
            throw std::runtime_error("safetensors tensor counts differ");
        }
        std::uint64_t compared = 0;
        long double squared = 0.0L;
        float maximum = 0.0F;
        std::string worst_tensor;
        bool finite = true;
        for (const auto& [name, expected_tensor] : baseline) {
            const auto found = candidate.find(name);
            if (found == candidate.end() ||
                found->second.shape() != expected_tensor.shape() ||
                found->second.dtype() != microllm::DType::Float32 ||
                expected_tensor.dtype() != microllm::DType::Float32) {
                throw std::runtime_error(
                    "safetensors name, shape, or dtype differs: " + name);
            }
            const auto expected = expected_tensor.to_vector();
            const auto actual = found->second.to_vector();
            for (std::size_t index = 0; index < expected.size(); ++index) {
                finite = finite && std::isfinite(expected[index]) &&
                         std::isfinite(actual[index]);
                const auto difference = std::abs(expected[index] - actual[index]);
                if (difference > maximum) {
                    maximum = difference;
                    worst_tensor = name;
                }
                squared += static_cast<long double>(difference) * difference;
            }
            if (expected.size() >
                std::numeric_limits<std::uint64_t>::max() - compared) {
                throw std::overflow_error("compared element count overflows");
            }
            compared += static_cast<std::uint64_t>(expected.size());
        }
        if (compared == 0) throw std::runtime_error("no tensor values compared");
        const auto rms = std::sqrt(
            static_cast<double>(squared / static_cast<long double>(compared)));
        std::cout << std::setprecision(12)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"record_type\":\"safetensors_complete_comparison\""
                  << ",\"tensor_count\":" << baseline.size()
                  << ",\"compared_elements\":" << compared
                  << ",\"all_finite\":" << (finite ? "true" : "false")
                  << ",\"maximum_absolute_difference\":" << maximum
                  << ",\"rms_difference\":" << rms
                  << ",\"worst_tensor\":\"" << worst_tensor << "\"}\n";
        return finite ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "microllm_compare_safetensors: " << error.what() << '\n';
        return 1;
    }
}

