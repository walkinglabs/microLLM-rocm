#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <microllm/multi_gpu/communicator.h>
#include <microllm/ops/low_level.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace {

struct Options {
    std::int64_t payload_elements = 262144;
    std::int64_t compute_elements = 4194304;
    int warmup = 3;
    int repetitions = 20;
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
        if (name == "--payload-elements") options.payload_elements = integer(argv[index + 1], "payload-elements");
        else if (name == "--compute-elements") options.compute_elements = integer(argv[index + 1], "compute-elements");
        else if (name == "--warmup") options.warmup = static_cast<int>(integer(argv[index + 1], "warmup"));
        else if (name == "--repetitions") options.repetitions = static_cast<int>(integer(argv[index + 1], "repetitions"));
        else throw std::invalid_argument("unknown option: " + std::string(name));
    }
    if (options.payload_elements <= 0 || options.compute_elements <= 0 || options.warmup < 0 ||
        options.repetitions <= 0 || options.payload_elements > 100000000 ||
        options.compute_elements > 100000000) {
        throw std::invalid_argument("overlap benchmark options are outside safety limits");
    }
    if (microllm::runtime::hip_device_count() < 2) {
        throw std::runtime_error("overlap benchmark requires two visible HIP devices");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        microllm::multi_gpu::Communicator communicator({0, 1});
        std::vector<microllm::runtime::Stream> compute_streams;
        std::vector<microllm::Tensor> compute_left;
        std::vector<microllm::Tensor> compute_right;
        std::vector<microllm::Tensor> compute_output;
        std::vector<microllm::Tensor> communication;
        for (int rank = 0; rank < 2; ++rank) {
            const auto device = microllm::Device::hip(rank);
            compute_streams.emplace_back(device);
            compute_left.emplace_back(microllm::Shape{options.compute_elements},
                                      microllm::DType::Float32, device);
            compute_right.emplace_back(microllm::Shape{options.compute_elements},
                                       microllm::DType::Float32, device);
            compute_output.emplace_back(microllm::Shape{options.compute_elements},
                                        microllm::DType::Float32, device);
            communication.emplace_back(microllm::Shape{options.payload_elements},
                                       microllm::DType::Float32, device);
            const microllm::ops::OpContext compute_context{&compute_streams.back(), nullptr, 0};
            const microllm::ops::OpContext communication_context{
                &communicator.stream(static_cast<std::size_t>(rank)), nullptr, 0};
            microllm::ops::fill_(compute_left.back(), 1.0F, compute_context);
            microllm::ops::fill_(compute_right.back(), 2.0F, compute_context);
            microllm::ops::fill_(communication.back(), static_cast<float>(rank + 1),
                                 communication_context);
        }
        auto enqueue_compute = [&] {
            for (std::size_t rank = 0; rank < compute_streams.size(); ++rank) {
                const microllm::ops::OpContext context{&compute_streams[rank], nullptr, 0};
                microllm::ops::add_out(compute_output[rank].view(),
                                        std::as_const(compute_left[rank]).view(),
                                        std::as_const(compute_right[rank]).view(), context);
            }
        };
        auto synchronize_compute = [&] {
            for (const auto& stream : compute_streams) stream.synchronize();
        };
        synchronize_compute();
        communicator.synchronize();
        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            enqueue_compute();
            synchronize_compute();
            communicator.all_reduce(communication, false);
        }

        const auto serialized_start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < options.repetitions; ++iteration) {
            enqueue_compute();
            synchronize_compute();
            communicator.enqueue_all_reduce_sum(communication);
            communicator.synchronize();
        }
        const auto serialized_finish = std::chrono::steady_clock::now();

        const auto overlapped_start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < options.repetitions; ++iteration) {
            enqueue_compute();
            communicator.enqueue_all_reduce_sum(communication);
            synchronize_compute();
            communicator.synchronize();
        }
        const auto overlapped_finish = std::chrono::steady_clock::now();
        const auto serialized_ms =
            std::chrono::duration<double, std::milli>(serialized_finish - serialized_start).count() /
            static_cast<double>(options.repetitions);
        const auto overlapped_ms =
            std::chrono::duration<double, std::milli>(overlapped_finish - overlapped_start).count() /
            static_cast<double>(options.repetitions);
        const auto overlap_gain = (serialized_ms - overlapped_ms) / serialized_ms;

        // Reset once and prove compute/communication results after timing.
        for (std::size_t rank = 0; rank < communication.size(); ++rank) {
            const microllm::ops::OpContext context{&communicator.stream(rank), nullptr, 0};
            microllm::ops::fill_(communication[rank], static_cast<float>(rank + 1), context);
        }
        communicator.synchronize();
        communicator.all_reduce(communication, false);
        const auto compute_guard = compute_output[0].to_vector()[0];
        const auto communication_guard = communication[0].to_vector()[0];
        const auto info = microllm::runtime::device_info(microllm::Device::hip());
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"engine_version\":\"" << MICROLLM_VERSION
                  << "\",\"operation\":\"compute_allreduce_overlap\""
                  << ",\"device_name\":\"" << info.name
                  << "\",\"architecture\":\"" << info.architecture
                  << "\",\"ranks\":2,\"payload_elements\":" << options.payload_elements
                  << ",\"compute_elements\":" << options.compute_elements
                  << ",\"warmup\":" << options.warmup
                  << ",\"repetitions\":" << options.repetitions
                  << ",\"serialized_step_ms\":" << serialized_ms
                  << ",\"overlapped_step_ms\":" << overlapped_ms
                  << ",\"overlap_gain\":" << overlap_gain
                  << ",\"compute_guard\":" << compute_guard
                  << ",\"communication_guard\":" << communication_guard << "}\n";
        return std::isfinite(overlap_gain) && compute_guard == 3.0F &&
                       communication_guard == 3.0F
                   ? 0
                   : 2;
    } catch (const std::exception& error) {
        std::cerr << "benchmark_overlap: " << error.what() << '\n';
        return 1;
    }
}
