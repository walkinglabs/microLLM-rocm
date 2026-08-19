#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <microllm/multi_gpu/communicator.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    int ranks = 2;
    std::int64_t total_elements = 262144;
    std::int64_t bucket_elements = 262144;
    int warmup = 3;
    int repetitions = 10;
};

std::int64_t integer(const char* value, const char* name) {
    char* end = nullptr;
    const auto parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0') throw std::invalid_argument(std::string("invalid ") + name);
    return parsed;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("option is missing a value");
        const std::string_view name(argv[index]);
        if (name == "--ranks") options.ranks = static_cast<int>(integer(argv[index + 1], "ranks"));
        else if (name == "--total-elements") options.total_elements = integer(argv[index + 1], "total-elements");
        else if (name == "--bucket-elements") options.bucket_elements = integer(argv[index + 1], "bucket-elements");
        else if (name == "--warmup") options.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        else if (name == "--repetitions") options.repetitions = static_cast<int>(integer(argv[index + 1], "repetitions"));
        else throw std::invalid_argument("unknown option: " + std::string(name));
    }
    if (options.ranks <= 0 || options.total_elements <= 0 || options.bucket_elements <= 0 ||
        options.warmup < 0 || options.repetitions <= 0) {
        throw std::invalid_argument("benchmark options are outside valid ranges");
    }
    if (options.ranks > microllm::runtime::hip_device_count()) {
        throw std::invalid_argument("requested ranks exceed visible HIP devices");
    }
    if (options.total_elements > 100000000) {
        throw std::invalid_argument("all-reduce payload exceeds safety limit");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        std::vector<int> devices(static_cast<std::size_t>(options.ranks));
        for (int rank = 0; rank < options.ranks; ++rank) devices[static_cast<std::size_t>(rank)] = rank;
        const auto init_start = std::chrono::steady_clock::now();
        microllm::multi_gpu::Communicator communicator(devices);
        const auto init_finish = std::chrono::steady_clock::now();
        const auto initialization_seconds =
            std::chrono::duration<double>(init_finish - init_start).count();

        std::vector<std::int64_t> bucket_sizes;
        for (std::int64_t offset = 0; offset < options.total_elements;
             offset += options.bucket_elements) {
            bucket_sizes.push_back(std::min(options.bucket_elements,
                                            options.total_elements - offset));
        }
        std::vector<std::vector<microllm::Tensor>> buckets(bucket_sizes.size());
        for (std::size_t bucket = 0; bucket < bucket_sizes.size(); ++bucket) {
            buckets[bucket].reserve(static_cast<std::size_t>(options.ranks));
            for (int rank = 0; rank < options.ranks; ++rank) {
                buckets[bucket].emplace_back(microllm::Shape{bucket_sizes[bucket]},
                                             microllm::DType::Float32,
                                             microllm::Device::hip(rank));
                microllm::ops::fill_(buckets[bucket].back(), static_cast<float>(rank + 1));
            }
        }
        auto reduce_all = [&] {
            for (auto& bucket : buckets) communicator.all_reduce(bucket, true);
        };
        for (int iteration = 0; iteration < options.warmup; ++iteration) reduce_all();
        const auto measured_start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < options.repetitions; ++iteration) reduce_all();
        const auto measured_finish = std::chrono::steady_clock::now();
        const auto measured_seconds =
            std::chrono::duration<double>(measured_finish - measured_start).count();
        const auto step_milliseconds = measured_seconds * 1000.0 /
                                       static_cast<double>(options.repetitions);
        const auto payload_bytes = static_cast<double>(options.total_elements) * sizeof(float);
        const auto algorithmic_gigabytes_per_second =
            payload_bytes * 2.0 * static_cast<double>(options.ranks - 1) /
            static_cast<double>(options.ranks) /
            (measured_seconds / static_cast<double>(options.repetitions)) / 1.0e9;
        const auto expected = static_cast<float>(options.ranks + 1) / 2.0F;
        float maximum_error = 0.0F;
        for (const auto& bucket : buckets) {
            for (const auto& rank_tensor : bucket) {
                const auto values = rank_tensor.to_vector();
                for (const auto value : values) {
                    maximum_error = std::max(maximum_error, std::abs(value - expected));
                }
            }
        }
        const auto info = microllm::runtime::device_info(microllm::Device::hip());
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"engine_version\":\"" << MICROLLM_VERSION
                  << "\",\"operation\":\"all_reduce_average\""
                  << ",\"device_name\":\"" << info.name
                  << "\",\"architecture\":\"" << info.architecture
                  << "\",\"ranks\":" << options.ranks
                  << ",\"total_elements\":" << options.total_elements
                  << ",\"total_bytes\":" << static_cast<std::uint64_t>(payload_bytes)
                  << ",\"bucket_elements\":" << options.bucket_elements
                  << ",\"bucket_count\":" << bucket_sizes.size()
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"initialization_seconds\":" << initialization_seconds
                  << ",\"step_milliseconds\":" << step_milliseconds
                  << ",\"algorithmic_gigabytes_per_second\":"
                  << algorithmic_gigabytes_per_second
                  << ",\"maximum_absolute_error\":" << maximum_error << "}\n";
        return maximum_error == 0.0F ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_allreduce: " << error.what() << '\n';
        return 1;
    }
}
