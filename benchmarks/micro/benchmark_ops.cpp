#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::string operation = "add";
    std::string device = "cpu";
    std::int64_t size = 256;
    int warmup = 5;
    int repetitions = 20;
};

std::int64_t parse_integer(const char* value, const char* name) {
    char* end = nullptr;
    const auto parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0') throw std::invalid_argument(std::string("invalid ") + name);
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("benchmark option is missing a value");
        const std::string_view name(argv[index]);
        if (name == "--op") options.operation = argv[index + 1];
        else if (name == "--device") options.device = argv[index + 1];
        else if (name == "--size") options.size = parse_integer(argv[index + 1], "size");
        else if (name == "--warmup") options.warmup = static_cast<int>(parse_integer(argv[index + 1], "warmup"));
        else if (name == "--repetitions") options.repetitions = static_cast<int>(parse_integer(argv[index + 1], "repetitions"));
        else throw std::invalid_argument("unknown benchmark option: " + std::string(name));
    }
    if (options.operation != "add" && options.operation != "matmul" &&
        options.operation != "softmax") {
        throw std::invalid_argument("op must be add, matmul, or softmax");
    }
    if (options.device != "cpu" && options.device != "hip") {
        throw std::invalid_argument("device must be cpu or hip");
    }
    if (options.size <= 0 || options.warmup < 0 || options.repetitions <= 0) {
        throw std::invalid_argument("size/repetition options are outside valid ranges");
    }
    if ((options.operation == "matmul" || options.operation == "softmax") &&
        options.size > 16384) {
        throw std::invalid_argument("matrix benchmark size exceeds the safety limit");
    }
    if (options.operation == "add" && options.size > 100000000) {
        throw std::invalid_argument("elementwise benchmark size exceeds the safety limit");
    }
    return options;
}

microllm::Tensor run_operation(const std::string& operation, const microllm::Tensor& left,
                               const microllm::Tensor& right,
                               const microllm::ops::OpContext& context = {}) {
    if (operation == "add") return microllm::ops::add(left, right, context);
    if (operation == "matmul") return microllm::ops::matmul(left, right, context);
    return microllm::ops::softmax(left, -1, context);
}

struct Summary {
    double minimum = 0.0;
    double mean = 0.0;
    double maximum = 0.0;
};

Summary summarize(const std::vector<double>& values) {
    if (values.empty()) throw std::invalid_argument("cannot summarize empty timings");
    return {*std::min_element(values.begin(), values.end()),
            std::accumulate(values.begin(), values.end(), 0.0) /
                static_cast<double>(values.size()),
            *std::max_element(values.begin(), values.end())};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto device = options.device == "cpu" ? microllm::Device::cpu()
                                                     : microllm::Device::hip();
        if (device.is_hip() && microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("HIP benchmark requested without a visible GPU");
        }
        const auto element_count = options.operation == "add" ? options.size
                                                                : options.size * options.size;
        std::vector<float> left_values(static_cast<std::size_t>(element_count));
        std::vector<float> right_values(static_cast<std::size_t>(element_count));
        for (std::size_t index = 0; index < left_values.size(); ++index) {
            left_values[index] = static_cast<float>(static_cast<int>(index % 29) - 14) / 29.0F;
            right_values[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 17.0F;
        }
        const microllm::Shape shape = options.operation == "add"
                                           ? microllm::Shape{options.size}
                                           : microllm::Shape{options.size, options.size};
        const auto left_cpu = microllm::Tensor::from_vector(left_values, shape);
        const auto right_cpu = microllm::Tensor::from_vector(right_values, shape);
        const auto reference = run_operation(options.operation, left_cpu, right_cpu).to_vector();
        const auto left = left_cpu.to(device);
        const auto right = right_cpu.to(device);
        const auto memory_before = microllm::runtime::memory_info(device);

        std::vector<double> wall_times;
        std::vector<double> kernel_times;
        microllm::Tensor output;
        if (device.is_hip()) {
            microllm::runtime::Stream stream(device);
            microllm::runtime::Event start(device);
            microllm::runtime::Event finish(device);
            const microllm::ops::OpContext context{&stream, nullptr, 0};
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                output = run_operation(options.operation, left, right, context);
            }
            stream.synchronize();
            for (int iteration = 0; iteration < options.repetitions; ++iteration) {
                const auto wall_start = std::chrono::steady_clock::now();
                start.record(stream);
                output = run_operation(options.operation, left, right, context);
                finish.record(stream);
                finish.synchronize();
                const auto wall_finish = std::chrono::steady_clock::now();
                kernel_times.push_back(finish.elapsed_ms_since(start));
                wall_times.push_back(std::chrono::duration<double, std::milli>(
                                         wall_finish - wall_start)
                                         .count());
            }
        } else {
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                output = run_operation(options.operation, left, right);
            }
            for (int iteration = 0; iteration < options.repetitions; ++iteration) {
                const auto start = std::chrono::steady_clock::now();
                output = run_operation(options.operation, left, right);
                const auto finish = std::chrono::steady_clock::now();
                const auto elapsed =
                    std::chrono::duration<double, std::milli>(finish - start).count();
                wall_times.push_back(elapsed);
                kernel_times.push_back(elapsed);
            }
        }

        const auto actual = output.to_vector();
        const auto memory_after = microllm::runtime::memory_info(device);
        float maximum_error = 0.0F;
        double value_sum = 0.0;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            maximum_error = std::max(maximum_error, std::abs(actual[index] - reference[index]));
            value_sum += actual[index];
        }
        const auto wall = summarize(wall_times);
        const auto kernel = summarize(kernel_times);
        const auto device_name = device.is_cpu() ? std::string("host CPU")
                                                  : microllm::runtime::device_info(device).name;
        const auto architecture = device.is_cpu() ? std::string("host")
                                                   : microllm::runtime::device_info(device).architecture;
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1"
                  << ",\"engine_version\":\"" << MICROLLM_VERSION << "\""
                  << ",\"op\":\"" << options.operation << "\""
                  << ",\"implementation\":\"readable\""
                  << ",\"device\":\"" << options.device << "\""
                  << ",\"device_name\":\"" << device_name << "\""
                  << ",\"architecture\":\"" << architecture << "\""
                  << ",\"hip_runtime_version\":" << microllm::runtime::hip_runtime_version()
                  << ",\"hip_driver_version\":" << microllm::runtime::hip_driver_version()
                  << ",\"dtype\":\"float32\""
                  << ",\"size\":" << options.size
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"kernel_ms_min\":" << kernel.minimum
                  << ",\"kernel_ms_mean\":" << kernel.mean
                  << ",\"kernel_ms_max\":" << kernel.maximum
                  << ",\"wall_ms_min\":" << wall.minimum
                  << ",\"wall_ms_mean\":" << wall.mean
                  << ",\"wall_ms_max\":" << wall.maximum
                  << ",\"gpu_free_bytes_before\":" << memory_before.free_bytes
                  << ",\"gpu_free_bytes_after\":" << memory_after.free_bytes
                  << ",\"gpu_total_bytes\":" << memory_after.total_bytes
                  << ",\"maximum_absolute_error\":" << maximum_error
                  << ",\"value_sum\":" << value_sum << "}\n";
        return maximum_error <= 2.0e-4F ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_ops: " << error.what() << '\n';
        return 1;
    }
}
