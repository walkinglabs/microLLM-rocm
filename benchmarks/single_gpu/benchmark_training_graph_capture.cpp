#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <hip/hip_runtime_api.h>

#include <microllm/autograd/autograd.h>
#include <microllm/core/tensor.h>
#include <microllm/model/model.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/optimizer.h>

namespace {

struct Options {
    std::string stage = "full-step";
    std::string precision = "fp32";
    std::size_t maximum_blocks = 65536;
};

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string_view name(argv[index]);
        if (name == "--stage") result.stage = argv[index + 1];
        else if (name == "--precision") result.precision = argv[index + 1];
        else if (name == "--maximum-blocks") {
            result.maximum_blocks =
                static_cast<std::size_t>(std::stoull(argv[index + 1]));
        } else {
            throw std::invalid_argument("unknown option: " + std::string(name));
        }
    }
    if ((result.stage != "forward" && result.stage != "backward" &&
         result.stage != "optimizer" && result.stage != "full-step") ||
        (result.precision != "fp32" && result.precision != "bf16") ||
        result.maximum_blocks == 0) {
        throw std::invalid_argument("training graph capture options are invalid");
    }
    return result;
}

std::string escaped(const std::string& value) {
    std::string result;
    for (const auto character : value) {
        if (character == '"' || character == '\\') result.push_back('\\');
        if (character == '\n') result += "\\n";
        else result.push_back(character);
    }
    return result;
}

microllm::model::ModelConfig tiny_config(const std::string& precision) {
    auto config = microllm::model::ModelConfig{
        .vocabulary_size = 64,
        .dimension = 32,
        .layers = 2,
        .heads = 4,
        .kv_heads = 2,
        .ffn_dimension = 64,
        .max_sequence_length = 16,
        .rope_base = 10000.0F,
        .tie_embeddings = false,
    };
    if (precision == "bf16") {
        config.linear_precision =
            microllm::model::LinearPrecision::BFloat16;
    }
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        if (!microllm::runtime::hip_compiled() ||
            microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("training graph probe requires HIP");
        }
        const auto device = microllm::Device::hip(0);
        microllm::model::TransformerModel model(
            tiny_config(command.precision), 17);
        model.to(device);
        microllm::model::Bf16TrainingMirrors mirrors;
        if (command.precision == "bf16") {
            mirrors = model.prepare_bf16_training_mirrors();
        }
        microllm::training::AdamW optimizer(
            model.parameters(),
            {.learning_rate = 1.0e-3F,
             .beta1 = 0.9F,
             .beta2 = 0.99F,
             .epsilon = 1.0e-8F,
             .weight_decay = 0.0F},
            mirrors);
        auto inputs = microllm::Tensor::from_int32_vector(
            {1, 2, 3, 4, 5, 6, 7, 8}, {1, 8}).to(device);
        auto targets = microllm::Tensor::from_int32_vector(
            {2, 3, 4, 5, 6, 7, 8, 9}, {1, 8}).to(device);
        microllm::runtime::Stream stream(device);
        microllm::runtime::ScopedDeferredHipStream scope(
            stream, command.maximum_blocks);
        std::optional<microllm::autograd::Value> loss;

        if (command.stage == "backward" || command.stage == "optimizer") {
            loss.emplace(model.loss(inputs, targets));
            if (command.stage == "optimizer") loss->backward();
            stream.synchronize();
        }

        microllm::runtime::HipGraphExecutable graph;
        bool capture_supported = false;
        int capture_status_after_failure =
            static_cast<int>(hipStreamCaptureStatusNone);
        int recovery_status = static_cast<int>(hipSuccess);
        int capture_status_after_recovery =
            static_cast<int>(hipStreamCaptureStatusNone);
        std::string capture_error;
        std::uint64_t optimizer_step_after_capture = optimizer.step_count();
        std::uint64_t optimizer_step_after_replay = optimizer.step_count();
        try {
            graph = microllm::runtime::HipGraphExecutable::capture(
                stream, [&] {
                    if (command.stage == "forward") {
                        loss.emplace(model.loss(inputs, targets));
                    } else if (command.stage == "backward") {
                        loss->backward();
                    } else if (command.stage == "optimizer") {
                        optimizer.step();
                    } else {
                        optimizer.zero_grad();
                        loss.emplace(model.loss(inputs, targets));
                        loss->backward();
                        optimizer.step();
                    }
                });
            capture_supported = true;
            optimizer_step_after_capture = optimizer.step_count();
            graph.launch(stream);
            stream.synchronize();
            optimizer_step_after_replay = optimizer.step_count();
        } catch (const std::exception& error) {
            capture_error = error.what();
            auto native_stream =
                reinterpret_cast<hipStream_t>(stream.native_handle());
            hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
            (void)hipStreamIsCapturing(native_stream, &status);
            capture_status_after_failure = static_cast<int>(status);
            if (status != hipStreamCaptureStatusNone) {
                hipGraph_t abandoned = nullptr;
                const auto recovery = hipStreamEndCapture(native_stream, &abandoned);
                recovery_status = static_cast<int>(recovery);
                if (abandoned != nullptr) (void)hipGraphDestroy(abandoned);
                (void)hipGetLastError();
            }
            status = hipStreamCaptureStatusNone;
            (void)hipStreamIsCapturing(native_stream, &status);
            capture_status_after_recovery = static_cast<int>(status);
        }
        const auto graph_nodes = graph.defined() ? graph.node_count() : 0U;
        const auto optimizer_replay_advances_host_step =
            optimizer_step_after_replay > optimizer_step_after_capture;
        const auto capture_recovery_failed =
            capture_status_after_recovery !=
            static_cast<int>(hipStreamCaptureStatusNone);
        if (!capture_recovery_failed) {
            graph = microllm::runtime::HipGraphExecutable();
            loss.reset();
            scope.finish();
        }
        const auto info = microllm::runtime::device_info(device);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"stage\":\"" << command.stage << "\""
                  << ",\"precision\":\"" << command.precision << "\""
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"capture_supported\":"
                  << (capture_supported ? "true" : "false")
                  << ",\"captured_nodes\":" << graph_nodes
                  << ",\"capture_error\":\"" << escaped(capture_error) << "\""
                  << ",\"capture_status_after_failure\":"
                  << capture_status_after_failure
                  << ",\"recovery_status\":" << recovery_status
                  << ",\"capture_status_after_recovery\":"
                  << capture_status_after_recovery
                  << ",\"capture_recovery_failed\":"
                  << (capture_recovery_failed ? "true" : "false")
                  << ",\"optimizer_step_after_capture\":"
                  << optimizer_step_after_capture
                  << ",\"optimizer_step_after_replay\":"
                  << optimizer_step_after_replay
                  << ",\"optimizer_replay_advances_host_step\":"
                  << (optimizer_replay_advances_host_step ? "true" : "false")
                  << ",\"deferred_blocks\":" << scope.total_deferred_blocks()
                  << ",\"deferred_bytes\":" << scope.total_deferred_bytes()
                  << "}\n";
        std::cout.flush();
        if (capture_recovery_failed) std::_Exit(EXIT_SUCCESS);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_bench_training_graph_capture: "
                  << error.what() << '\n';
        return 1;
    }
}
