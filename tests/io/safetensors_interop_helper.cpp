#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <microllm/io/safetensors.h>

namespace {

microllm::io::StateDict fixture() {
    return {
        {"layer.weight", microllm::Tensor::from_vector(
                             {-3.25F, -1.0F, -0.125F, 0.0F, 1.5F, 7.75F}, {2, 3})},
        {"norm.weight", microllm::Tensor::from_vector({1.0F, 0.5F, 2.0F}, {3})},
        {"scalar", microllm::Tensor::from_vector({0.333251953125F}, {})},
    };
}

void expect_fixture(const std::filesystem::path& path, float tolerance) {
    const auto actual = microllm::io::load_safetensors(path);
    const auto expected = fixture();
    if (actual.size() != expected.size()) throw std::runtime_error("tensor count mismatch");
    for (const auto& [name, expected_tensor] : expected) {
        const auto iterator = actual.find(name);
        if (iterator == actual.end()) throw std::runtime_error("missing tensor: " + name);
        if (iterator->second.shape() != expected_tensor.shape()) {
            throw std::runtime_error("shape mismatch: " + name);
        }
        const auto actual_values = iterator->second.to_vector();
        const auto expected_values = expected_tensor.to_vector();
        for (std::size_t index = 0; index < actual_values.size(); ++index) {
            if (std::abs(actual_values[index] - expected_values[index]) > tolerance) {
                throw std::runtime_error("value mismatch: " + name + " index=" +
                                         std::to_string(index));
            }
        }
    }
}

microllm::io::StateDict int8_fixture() {
    return {
        {"linear.weight", microllm::Tensor::from_int8_vector(
                              {-127, -3, -1, 0, 2, 126}, {2, 3})},
        {"linear.weight.scale",
         microllm::Tensor::from_vector({0.03125F}, {})},
    };
}

void expect_int8_fixture(const std::filesystem::path& path) {
    const auto actual = microllm::io::load_safetensors(path);
    const auto expected = int8_fixture();
    if (actual.size() != expected.size()) {
        throw std::runtime_error("INT8 tensor count mismatch");
    }
    if (actual.at("linear.weight").dtype() != microllm::DType::Int8 ||
        actual.at("linear.weight").shape() != microllm::Shape{2, 3} ||
        actual.at("linear.weight").to_int8_vector() !=
            expected.at("linear.weight").to_int8_vector()) {
        throw std::runtime_error("INT8 weight mismatch");
    }
    if (actual.at("linear.weight.scale").dtype() !=
            microllm::DType::Float32 ||
        actual.at("linear.weight.scale").to_vector() !=
            std::vector<float>{0.03125F}) {
        throw std::runtime_error("INT8 scale mismatch");
    }
}

void write_fixtures(const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    const auto state = fixture();
    using microllm::io::WeightFileDType;
    for (const auto& [name, dtype] :
         {std::tuple{"f32", WeightFileDType::Float32},
          std::tuple{"bf16", WeightFileDType::BFloat16},
          std::tuple{"f16", WeightFileDType::Float16}}) {
        microllm::io::save_safetensors(
            directory / (std::string("cpp_") + name + ".safetensors"), state,
            {.dtype = dtype, .atomic_replace = true});
    }
    microllm::io::save_safetensors(
        directory / "cpp_i8.safetensors", int8_fixture(),
        {.dtype = WeightFileDType::Preserve, .atomic_replace = true});
}

void verify_python_fixtures(const std::filesystem::path& directory) {
    expect_fixture(directory / "python_f32.safetensors", 0.0F);
    expect_fixture(directory / "python_bf16.safetensors", 2.0e-2F);
    expect_fixture(directory / "python_f16.safetensors", 2.0e-3F);
    expect_int8_fixture(directory / "python_i8.safetensors");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::invalid_argument("usage: helper <write|verify> <directory>");
        }
        const std::string mode = argv[1];
        const std::filesystem::path directory = argv[2];
        if (mode == "write") write_fixtures(directory);
        else if (mode == "verify") verify_python_fixtures(directory);
        else throw std::invalid_argument("mode must be write or verify");
        std::cout << "mode=" << mode << "\nstatus=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_safetensors_interop: " << error.what() << '\n';
        return 1;
    }
}
