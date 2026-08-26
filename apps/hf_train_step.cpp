#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <microllm/model/huggingface.h>
#include <microllm/model/model.h>
#include <microllm/autograd/diagnostics.h>
#include <microllm/io/safetensors.h>
#include <microllm/runtime/diagnostics.h>
#include <microllm/runtime/memory.h>
#include <microllm/runtime/runtime.h>
#include <microllm/training/optimizer.h>

namespace {

struct StepResult {
    float loss = 0.0F;
    double optimizer_ms = 0.0;
    microllm::runtime::TransferStats transfers;
};

struct Bf16AlgorithmRegistration {
    std::int64_t rows = 0;
    std::int64_t inner = 0;
    std::int64_t columns = 0;
    int index = -1;
};

std::vector<Bf16AlgorithmRegistration> parse_bf16_algorithms(
    const std::string& text) {
    if (text.empty()) return {};
    std::vector<Bf16AlgorithmRegistration> result;
    std::stringstream records(text);
    std::string record;
    while (std::getline(records, record, ',')) {
        std::stringstream fields(record);
        std::string field;
        std::vector<std::int64_t> values;
        while (std::getline(fields, field, ':')) {
            if (field.empty()) throw std::invalid_argument("empty BF16 algorithm field");
            std::int64_t value = 0;
            const auto parsed = std::from_chars(
                field.data(), field.data() + field.size(), value);
            if (parsed.ec != std::errc{} ||
                parsed.ptr != field.data() + field.size()) {
                throw std::invalid_argument(
                    "BF16 algorithms must be rows:inner:columns:index records");
            }
            values.push_back(value);
        }
        if (values.size() != 4 || values[0] <= 0 || values[1] <= 0 ||
            values[2] <= 0 || values[3] < 0 ||
            values[3] > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(
                "BF16 algorithms must be rows:inner:columns:index records");
        }
        result.push_back({values[0], values[1], values[2],
                          static_cast<int>(values[3])});
    }
    return result;
}

std::vector<std::int32_t> parse_tokens(std::string_view text) {
    std::vector<std::int32_t> output;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto item = text.substr(0, comma);
        std::int32_t value = 0;
        const auto parsed = std::from_chars(item.data(), item.data() + item.size(), value);
        if (item.empty() || parsed.ec != std::errc{} ||
            parsed.ptr != item.data() + item.size() || value < 0) {
            throw std::invalid_argument("tokens must be comma-separated nonnegative IDs");
        }
        output.push_back(value);
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    if (output.size() < 2) throw std::invalid_argument("training requires at least two tokens");
    return output;
}

void write_diagnostics(
    const std::filesystem::path& path,
    const microllm::autograd::GradientAccumulationDiagnostics& accumulation,
    const microllm::runtime::StridedCopyDiagnostics& strided) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open diagnostics output");
    output << "{\"schema_version\":1,\"status\":\"pass\""
           << ",\"gradient_accumulation\":{\"first_assignments\":"
           << accumulation.first_assignments
           << ",\"add_calls\":" << accumulation.add_calls
           << ",\"materializations\":" << accumulation.materializations
           << ",\"sparse_embedding_add_calls\":"
           << accumulation.sparse_embedding_add_calls
           << ",\"unique_dense_add_candidates\":"
           << accumulation.unique_dense_add_candidates
           << ",\"unique_dense_add_executed\":"
           << accumulation.unique_dense_add_executed
           << ",\"added_elements\":" << accumulation.added_elements
           << ",\"unique_dense_add_elements\":"
           << accumulation.unique_dense_add_elements
           << ",\"unique_dense_add_executed_elements\":"
           << accumulation.unique_dense_add_executed_elements
           << ",\"materialized_elements\":" << accumulation.materialized_elements
           << ",\"records\":[";
    for (std::size_t index = 0; index < accumulation.records.size(); ++index) {
        if (index != 0) output << ',';
        const auto& record = accumulation.records[index];
        output << "{\"target_operation\":\"" << record.target_operation
               << "\",\"first_source\":\"" << record.first_source
               << "\",\"last_add_source\":\"" << record.last_add_source
               << "\",\"shape\":[";
        for (std::size_t dimension = 0; dimension < record.shape.size(); ++dimension) {
            if (dimension != 0) output << ',';
            output << record.shape[dimension];
        }
        output << "],\"first_assignments\":" << record.first_assignments
               << ",\"add_calls\":" << record.add_calls
               << ",\"materializations\":" << record.materializations
               << ",\"sparse_embedding_add_calls\":"
               << record.sparse_embedding_add_calls
               << ",\"unique_dense_add_candidates\":"
               << record.unique_dense_add_candidates
               << ",\"unique_dense_add_executed\":"
               << record.unique_dense_add_executed
               << ",\"added_elements\":" << record.added_elements
               << ",\"unique_dense_add_elements\":"
               << record.unique_dense_add_elements
               << ",\"unique_dense_add_executed_elements\":"
               << record.unique_dense_add_executed_elements
               << ",\"materialized_elements\":" << record.materialized_elements
               << '}';
    }
    output << "]},\"strided_copy\":{\"calls\":" << strided.calls
           << ",\"elements\":" << strided.elements
           << ",\"bytes\":" << strided.bytes << ",\"records\":[";
    for (std::size_t index = 0; index < strided.records.size(); ++index) {
        if (index != 0) output << ',';
        const auto& record = strided.records[index];
        output << "{\"shape\":[";
        for (std::size_t dimension = 0; dimension < record.shape.size(); ++dimension) {
            if (dimension != 0) output << ',';
            output << record.shape[dimension];
        }
        output << "],\"strides\":[";
        for (std::size_t dimension = 0; dimension < record.strides.size(); ++dimension) {
            if (dimension != 0) output << ',';
            output << record.strides[dimension];
        }
        output << "],\"element_bytes\":" << record.element_bytes
               << ",\"device\":\"" << record.device.str()
               << "\",\"calls\":" << record.calls
               << ",\"elements\":" << record.elements
               << ",\"bytes\":" << record.bytes << '}';
    }
    output << "]}}\n";
    if (!output) throw std::runtime_error("cannot write diagnostics output");
}

void write_loss_trajectory(const std::filesystem::path& path,
                           const std::vector<float>& losses) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open loss trajectory output");
    output << std::setprecision(9)
           << "{\"schema_version\":1,\"status\":\"pass\",\"losses\":[";
    for (std::size_t index = 0; index < losses.size(); ++index) {
        if (index != 0) output << ',';
        output << losses[index];
    }
    output << "]}\n";
    if (!output) throw std::runtime_error("cannot write loss trajectory output");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path config_path;
        std::filesystem::path weights_path;
        std::string token_text;
        std::string device_text = "hip";
        float learning_rate = 1.0e-5F;
        int warmup = 0;
        int steps = 1;
        int batch_size = 1;
        std::string linear_precision = "fp32";
        std::string adamw_implementation = "auto";
        std::string adamw_moment_precision = "fp32";
        std::int64_t adamw_bf16_multi_tensor_threshold = -1;
        std::string bf16_algorithm_text;
        int fp32_gate_up_weight_gradient_solution_index = -1;
        std::filesystem::path diagnostics_output;
        std::filesystem::path loss_trajectory_output;
        std::filesystem::path gate_up_parameters_output;
        bool bf16_weight_mirrors = true;
        bool tied_embedding_sparse_add = true;
        bool unique_gradient_inplace_add = false;
        bool attention_rope_layout_fusion = true;
        bool attention_context_layout_fusion = true;
        bool attention_layout_plan_cache = false;
        bool attention_gemm_scale_fusion = false;
        bool attention_paired_gqa_repeat = false;
        bool attention_gqa_value_broadcast = false;
        bool attention_gqa_forward_value_broadcast = false;
        for (int index = 1; index < argc; index += 2) {
            if (index + 1 >= argc) throw std::invalid_argument("missing CLI value");
            const std::string name = argv[index];
            if (name == "--config") config_path = argv[index + 1];
            else if (name == "--weights") weights_path = argv[index + 1];
            else if (name == "--tokens") token_text = argv[index + 1];
            else if (name == "--device") device_text = argv[index + 1];
            else if (name == "--learning-rate") learning_rate = std::stof(argv[index + 1]);
            else if (name == "--warmup") warmup = std::stoi(argv[index + 1]);
            else if (name == "--steps") steps = std::stoi(argv[index + 1]);
            else if (name == "--batch") batch_size = std::stoi(argv[index + 1]);
            else if (name == "--linear-precision") linear_precision = argv[index + 1];
            else if (name == "--adamw-implementation") {
                adamw_implementation = argv[index + 1];
            }
            else if (name == "--adamw-moment-precision") {
                adamw_moment_precision = argv[index + 1];
            }
            else if (name == "--adamw-bf16-multi-tensor-threshold") {
                const std::string value = argv[index + 1];
                adamw_bf16_multi_tensor_threshold =
                    value == "auto" ? -1 : std::stoll(value);
            }
            else if (name == "--bf16-algorithms") {
                bf16_algorithm_text = argv[index + 1];
            }
            else if (name ==
                     "--fp32-gate-up-weight-gradient-solution-index") {
                fp32_gate_up_weight_gradient_solution_index =
                    std::stoi(argv[index + 1]);
            }
            else if (name == "--diagnostics-output") {
                diagnostics_output = argv[index + 1];
            }
            else if (name == "--loss-trajectory-output") {
                loss_trajectory_output = argv[index + 1];
            }
            else if (name == "--gate-up-parameters-output") {
                gate_up_parameters_output = argv[index + 1];
            }
            else if (name == "--tied-embedding-sparse-add") {
                const std::string value = argv[index + 1];
                if (value != "true" && value != "false") {
                    throw std::invalid_argument(
                        "--tied-embedding-sparse-add must be true or false");
                }
                tied_embedding_sparse_add = value == "true";
            }
            else if (name == "--unique-gradient-inplace-add") {
                const std::string value = argv[index + 1];
                if (value != "true" && value != "false") {
                    throw std::invalid_argument(
                        "--unique-gradient-inplace-add must be true or false");
                }
                unique_gradient_inplace_add = value == "true";
            }
            else if (name == "--attention-rope-layout-fusion") {
                const std::string value = argv[index + 1];
                if (value != "true" && value != "false") {
                    throw std::invalid_argument(
                        "--attention-rope-layout-fusion must be true or false");
                }
                attention_rope_layout_fusion = value == "true";
            }
            else if (name == "--attention-context-layout-fusion") {
                const std::string value = argv[index + 1];
                if (value != "true" && value != "false") {
                    throw std::invalid_argument(
                        "--attention-context-layout-fusion must be true or false");
                }
                attention_context_layout_fusion = value == "true";
            }
            else if (name == "--attention-layout-plan-cache") {
                const std::string value = argv[index + 1];
                if (value != "true" && value != "false") {
                    throw std::invalid_argument(
                        "--attention-layout-plan-cache must be true or false");
                }
                attention_layout_plan_cache = value == "true";
            }
            else if (name == "--attention-gemm-scale-fusion") {
                const std::string value = argv[index + 1];
                if (value != "true" && value != "false") {
                    throw std::invalid_argument(
                        "--attention-gemm-scale-fusion must be true or false");
                }
                attention_gemm_scale_fusion = value == "true";
            }
            else if (name == "--attention-paired-gqa-repeat") {
                const std::string value = argv[index + 1];
                if (value != "true" && value != "false") {
                    throw std::invalid_argument(
                        "--attention-paired-gqa-repeat must be true or false");
                }
                attention_paired_gqa_repeat = value == "true";
            }
            else if (name == "--attention-gqa-value-broadcast") {
                const std::string value = argv[index + 1];
                if (value != "true" && value != "false") {
                    throw std::invalid_argument(
                        "--attention-gqa-value-broadcast must be true or false");
                }
                attention_gqa_value_broadcast = value == "true";
            }
            else if (name == "--attention-gqa-forward-value-broadcast") {
                const std::string value = argv[index + 1];
                if (value != "true" && value != "false") {
                    throw std::invalid_argument(
                        "--attention-gqa-forward-value-broadcast must be true or false");
                }
                attention_gqa_forward_value_broadcast = value == "true";
            }
            else if (name == "--bf16-weight-mirrors") {
                const std::string value = argv[index + 1];
                if (value != "true" && value != "false") {
                    throw std::invalid_argument(
                        "--bf16-weight-mirrors must be true or false");
                }
                bf16_weight_mirrors = value == "true";
            }
            else throw std::invalid_argument("unknown option: " + name);
        }
        if (config_path.empty() || weights_path.empty() || token_text.empty()) {
            throw std::invalid_argument("--config, --weights, and --tokens are required");
        }
        if (warmup < 0 || steps <= 0 || batch_size <= 0) {
            throw std::invalid_argument(
                "--warmup must be nonnegative; --steps and --batch must be positive");
        }
        if (linear_precision != "fp32" && linear_precision != "bf16") {
            throw std::invalid_argument("--linear-precision must be fp32 or bf16");
        }
        if (adamw_implementation != "auto" && adamw_implementation != "scalar" &&
            adamw_implementation != "vectorized") {
            throw std::invalid_argument(
                "--adamw-implementation must be auto, scalar, or vectorized");
        }
        if (adamw_moment_precision != "fp32" &&
            adamw_moment_precision != "bf16") {
            throw std::invalid_argument(
                "--adamw-moment-precision must be fp32 or bf16");
        }
        if (adamw_moment_precision == "bf16" &&
            adamw_implementation == "vectorized") {
            throw std::invalid_argument(
                "BF16 AdamW moments cannot use the FP32 vectorized selector");
        }
        if (adamw_bf16_multi_tensor_threshold < -1 ||
            (adamw_bf16_multi_tensor_threshold > 0 &&
             (adamw_moment_precision != "bf16" || device_text != "hip"))) {
            throw std::invalid_argument(
                "--adamw-bf16-multi-tensor-threshold must be auto or non-negative "
                "and requires BF16 moments on HIP");
        }
        const auto bf16_algorithms = parse_bf16_algorithms(bf16_algorithm_text);
        microllm::autograd::enable_tied_embedding_sparse_add(
            tied_embedding_sparse_add);
        microllm::autograd::enable_unique_gradient_inplace_add(
            unique_gradient_inplace_add);
        microllm::autograd::enable_attention_rope_layout_fusion(
            attention_rope_layout_fusion);
        microllm::autograd::enable_attention_context_layout_fusion(
            attention_context_layout_fusion);
        microllm::ops::enable_attention_layout_plan_cache(
            attention_layout_plan_cache);
        microllm::ops::enable_attention_gemm_scale_fusion(
            attention_gemm_scale_fusion);
        microllm::ops::enable_attention_paired_gqa_repeat(
            attention_paired_gqa_repeat);
        microllm::ops::enable_attention_gqa_value_broadcast(
            attention_gqa_value_broadcast);
        microllm::ops::enable_attention_gqa_forward_value_broadcast(
            attention_gqa_forward_value_broadcast);
        if (!bf16_algorithms.empty() &&
            (linear_precision != "bf16" || device_text != "hip")) {
            throw std::invalid_argument(
                "--bf16-algorithms requires BF16 training on HIP");
        }
        if (fp32_gate_up_weight_gradient_solution_index < -1 ||
            (fp32_gate_up_weight_gradient_solution_index >= 0 &&
             (linear_precision != "bf16" || device_text != "hip"))) {
            throw std::invalid_argument(
                "--fp32-gate-up-weight-gradient-solution-index requires "
                "BF16 training on HIP and a non-negative index");
        }
        const auto device = device_text == "hip" ? microllm::Device::hip(0)
                                                   : microllm::Device::cpu();
        if (device_text != "cpu" && device_text != "hip") {
            throw std::invalid_argument("--device must be cpu or hip");
        }
        auto external = microllm::model::load_huggingface_config(config_path);
        if (linear_precision == "bf16") {
            external.model.linear_precision = microllm::model::LinearPrecision::BFloat16;
        }
        microllm::ops::clear_bf16_algorithm_registry();
        for (const auto& algorithm : bf16_algorithms) {
            microllm::ops::register_bf16_algorithm(
                algorithm.rows, algorithm.inner, algorithm.columns,
                microllm::DType::Float32, algorithm.index);
        }
        microllm::runtime::reset_allocation_peak(device);
        microllm::model::TransformerModel model(
            external.model, 1,
            microllm::model::ParameterInitialization::Uninitialized);
        model.to(device);
        microllm::model::LoadWeightsOptions load_options;
        load_options.mapping = microllm::model::qwen_style_weight_mapping(external.model);
        load_options.aliases =
            microllm::model::qwen3_tied_weight_aliases(external.model);
        const auto load_start = std::chrono::steady_clock::now();
        microllm::runtime::reset_transfer_stats();
        const auto report = model.load_safetensors(weights_path, load_options);
        microllm::runtime::synchronize(device);
        const auto load_finish = std::chrono::steady_clock::now();
        const auto load_transfers = microllm::runtime::transfer_stats();
        const auto load_allocation = microllm::runtime::allocation_stats(device);
        const auto load_ms = std::chrono::duration<double, std::milli>(
                                 load_finish - load_start).count();
        microllm::model::Bf16TrainingMirrors bf16_mirrors;
        if (linear_precision == "bf16" && bf16_weight_mirrors) {
            bf16_mirrors = model.prepare_bf16_training_mirrors();
        }
        microllm::training::AdamW optimizer(
            model.parameters(), {.learning_rate = learning_rate,
                                 .beta1 = 0.9F,
                                 .beta2 = 0.999F,
                                 .epsilon = 1.0e-8F,
                                 .weight_decay = 0.01F,
                                 .moment_precision =
                                     adamw_moment_precision == "bf16"
                                         ? microllm::training::AdamWConfig::
                                               MomentPrecision::BFloat16
                                         : microllm::training::AdamWConfig::
                                               MomentPrecision::Float32},
            bf16_mirrors,
            adamw_implementation == "scalar"
                ? microllm::ops::AdamWImplementation::Scalar
                : adamw_implementation == "vectorized"
                      ? microllm::ops::AdamWImplementation::Vectorized
                      : microllm::ops::AdamWImplementation::Auto,
            adamw_bf16_multi_tensor_threshold);
        const auto all_tokens = parse_tokens(token_text);
        const std::vector<std::int32_t> input_ids(all_tokens.begin(), all_tokens.end() - 1);
        const std::vector<std::int32_t> target_ids(all_tokens.begin() + 1, all_tokens.end());
        microllm::ops::clear_fp32_matmul_solution_registry();
        if (fp32_gate_up_weight_gradient_solution_index >= 0) {
            const auto rows = static_cast<std::int64_t>(input_ids.size()) *
                              batch_size;
            const auto key = microllm::ops::make_fp32_matmul_solution_key(
                {rows, external.model.dimension},
                {rows, external.model.ffn_dimension}, device,
                true, false);
            microllm::ops::register_fp32_matmul_solution(
                key, fp32_gate_up_weight_gradient_solution_index);
        }
        std::vector<std::int32_t> batched_inputs;
        std::vector<std::int32_t> batched_targets;
        batched_inputs.reserve(input_ids.size() * static_cast<std::size_t>(batch_size));
        batched_targets.reserve(target_ids.size() * static_cast<std::size_t>(batch_size));
        for (int batch = 0; batch < batch_size; ++batch) {
            batched_inputs.insert(batched_inputs.end(), input_ids.begin(), input_ids.end());
            batched_targets.insert(batched_targets.end(), target_ids.begin(), target_ids.end());
        }
        auto inputs = microllm::Tensor::from_int32_vector(
            batched_inputs, {batch_size, static_cast<std::int64_t>(input_ids.size())});
        auto targets = microllm::Tensor::from_int32_vector(
            batched_targets, {batch_size, static_cast<std::int64_t>(target_ids.size())});
        if (device.is_hip()) { inputs = inputs.to(device); targets = targets.to(device); }
        auto named = model.named_parameters();
        microllm::autograd::Value* observed = nullptr;
        for (const auto& [name, parameter] : named) {
            if (name == "final_norm.weight") observed = parameter;
        }
        if (observed == nullptr) throw std::logic_error("final_norm.weight is missing");
        const auto run_step = [&]() {
            optimizer.zero_grad();
            const auto loss = model.loss(inputs, targets);
            const auto loss_value = loss.data().to_vector()[0];
            loss.backward();
            // backward is asynchronous on HIP. Complete it before the host
            // timer starts so optimizer_ms measures optimizer work rather
            // than an arbitrary backward tail. End-to-end measured_ms still
            // spans both phases and therefore keeps the real step cost.
            microllm::runtime::synchronize(device);
            microllm::runtime::reset_transfer_stats();
            const auto optimizer_start = std::chrono::steady_clock::now();
            optimizer.step();
            microllm::runtime::synchronize(device);
            const auto optimizer_finish = std::chrono::steady_clock::now();
            return StepResult{
                loss_value,
                std::chrono::duration<double, std::milli>(optimizer_finish - optimizer_start)
                    .count(),
                microllm::runtime::transfer_stats()};
        };
        const auto warmup_start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < warmup; ++iteration) (void)run_step();
        microllm::runtime::synchronize(device);
        const auto warmup_finish = std::chrono::steady_clock::now();
        if (device.is_hip()) microllm::runtime::enable_hip_caching_allocator(device);
        microllm::runtime::reset_allocation_peak(device);
        if (!diagnostics_output.empty()) {
            microllm::autograd::reset_gradient_accumulation_diagnostics();
            microllm::runtime::reset_strided_copy_diagnostics();
            microllm::autograd::enable_gradient_accumulation_diagnostics(true);
            microllm::runtime::enable_strided_copy_diagnostics(true);
        }
        const auto before = observed->data().to_vector().front();
        float first_loss = 0.0F;
        float final_loss = 0.0F;
        double optimizer_ms = 0.0;
        std::vector<float> loss_trajectory;
        loss_trajectory.reserve(static_cast<std::size_t>(steps));
        std::uint64_t optimizer_h2d = 0;
        std::uint64_t optimizer_d2h = 0;
        std::uint64_t optimizer_h2d_bytes = 0;
        std::uint64_t optimizer_d2h_bytes = 0;
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < steps; ++iteration) {
            const auto result = run_step();
            if (iteration == 0) first_loss = result.loss;
            final_loss = result.loss;
            loss_trajectory.push_back(result.loss);
            optimizer_ms += result.optimizer_ms;
            optimizer_h2d += result.transfers.host_to_device_calls;
            optimizer_d2h += result.transfers.device_to_host_calls;
            optimizer_h2d_bytes += result.transfers.host_to_device_bytes;
            optimizer_d2h_bytes += result.transfers.device_to_host_bytes;
        }
        const auto finish = std::chrono::steady_clock::now();
        microllm::autograd::enable_gradient_accumulation_diagnostics(false);
        microllm::runtime::enable_strided_copy_diagnostics(false);
        if (!diagnostics_output.empty()) {
            write_diagnostics(
                diagnostics_output,
                microllm::autograd::gradient_accumulation_diagnostics(),
                microllm::runtime::strided_copy_diagnostics());
        }
        const auto after = observed->data().to_vector().front();
        const auto allocation = microllm::runtime::allocation_stats(device);
        const auto info = device.is_cpu()
                              ? microllm::runtime::DeviceInfo{device, "host CPU", "host"}
                              : microllm::runtime::device_info(device);
        const auto attention_plan_stats =
            microllm::ops::attention_layout_plan_cache_stats();
        const auto fp32_solution_stats =
            microllm::ops::fp32_matmul_solution_stats();
        const auto measured_ms =
            std::chrono::duration<double, std::milli>(finish - start).count();
        if (!loss_trajectory_output.empty()) {
            write_loss_trajectory(loss_trajectory_output, loss_trajectory);
        }
        std::size_t gate_up_parameter_tensors = 0;
        std::uint64_t gate_up_parameter_elements = 0;
        if (!gate_up_parameters_output.empty()) {
            microllm::io::StateDict selected;
            for (const auto& [name, parameter] : named) {
                if (!name.ends_with(".gate_proj.weight") &&
                    !name.ends_with(".up_proj.weight")) {
                    continue;
                }
                gate_up_parameter_elements += static_cast<std::uint64_t>(
                    parameter->data().numel());
                selected.emplace(name, parameter->data());
            }
            gate_up_parameter_tensors = selected.size();
            if (gate_up_parameter_tensors == 0) {
                throw std::logic_error("gate/up parameter selection is empty");
            }
            microllm::io::save_safetensors(
                gate_up_parameters_output, selected,
                {.dtype = microllm::io::WeightFileDType::Float32,
                 .atomic_replace = true});
        }
        const auto warmup_ms =
            std::chrono::duration<double, std::milli>(warmup_finish - warmup_start).count();
        const auto trained_tokens = input_ids.size() * static_cast<std::size_t>(batch_size) *
                                    static_cast<std::size_t>(steps);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"device\":\"" << device.str() << "\""
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"hip_runtime_version\":"
                  << microllm::runtime::hip_runtime_version()
                  << ",\"hip_driver_version\":" << microllm::runtime::hip_driver_version()
                  << ",\"compute_dtype\":\""
                  << (linear_precision == "bf16" ? "bf16_linear_fp32_master" : "float32")
                  << "\""
                  << ",\"bf16_weight_mirrors_enabled\":"
                  << (!bf16_mirrors.empty() ? "true" : "false")
                  << ",\"adamw_implementation\":\"" << adamw_implementation << "\""
                  << ",\"adamw_moment_precision\":\""
                  << adamw_moment_precision << "\""
                  << ",\"adamw_moment_state_bytes\":"
                  << optimizer.moment_state_bytes()
                  << ",\"adamw_multi_tensor_update\":"
                  << (optimizer.bf16_multi_tensor_count() > 0 ? "true" : "false")
                  << ",\"adamw_bf16_multi_tensor_threshold\":"
                  << optimizer.bf16_multi_tensor_threshold()
                  << ",\"adamw_bf16_multi_tensor_tensors\":"
                  << optimizer.bf16_multi_tensor_count()
                  << ",\"adamw_bf16_multi_tensor_elements\":"
                  << optimizer.bf16_multi_tensor_elements()
                  << ",\"bf16_algorithm_registrations\":"
                  << bf16_algorithms.size()
                  << ",\"bf16_algorithm_spec\":\"" << bf16_algorithm_text << "\""
                  << ",\"fp32_gate_up_weight_gradient_solution_index\":"
                  << fp32_gate_up_weight_gradient_solution_index
                  << ",\"fp32_solution_registered_entries\":"
                  << fp32_solution_stats.registered_entries
                  << ",\"fp32_solution_registry_hits\":"
                  << fp32_solution_stats.registry_hits
                  << ",\"fp32_solution_registry_misses\":"
                  << fp32_solution_stats.registry_misses
                  << ",\"fp32_solution_dispatches\":"
                  << fp32_solution_stats.dispatches
                  << ",\"diagnostics_enabled\":"
                  << (!diagnostics_output.empty() ? "true" : "false")
                  << ",\"loss_trajectory_output_written\":"
                  << (!loss_trajectory_output.empty() ? "true" : "false")
                  << ",\"loss_trajectory_steps\":" << loss_trajectory.size()
                  << ",\"gate_up_parameters_output_written\":"
                  << (!gate_up_parameters_output.empty() ? "true" : "false")
                  << ",\"gate_up_parameter_tensors\":"
                  << gate_up_parameter_tensors
                  << ",\"gate_up_parameter_elements\":"
                  << gate_up_parameter_elements
                  << ",\"tied_embedding_sparse_add\":"
                  << (tied_embedding_sparse_add ? "true" : "false")
                  << ",\"unique_gradient_inplace_add\":"
                  << (unique_gradient_inplace_add ? "true" : "false")
                  << ",\"attention_rope_layout_fusion\":"
                  << (attention_rope_layout_fusion ? "true" : "false")
                  << ",\"attention_context_layout_fusion\":"
                  << (attention_context_layout_fusion ? "true" : "false")
                  << ",\"attention_layout_plan_cache\":"
                  << (attention_layout_plan_cache ? "true" : "false")
                  << ",\"attention_layout_plan_cache_entries\":"
                  << attention_plan_stats.entries
                  << ",\"attention_layout_plan_cache_hits\":"
                  << attention_plan_stats.hits
                  << ",\"attention_layout_plan_cache_misses\":"
                  << attention_plan_stats.misses
                  << ",\"attention_gemm_scale_fusion\":"
                  << (attention_gemm_scale_fusion ? "true" : "false")
                  << ",\"attention_paired_gqa_repeat\":"
                  << (attention_paired_gqa_repeat ? "true" : "false")
                  << ",\"attention_gqa_value_broadcast\":"
                  << (attention_gqa_value_broadcast ? "true" : "false")
                  << ",\"attention_gqa_forward_value_broadcast\":"
                  << (attention_gqa_forward_value_broadcast ? "true" : "false")
                  << ",\"measurement_profile\":\""
                  << (warmup > 0 || steps > 1 ? "comparison" : "smoke") << "\""
                  << ",\"loaded_tensors\":" << report.loaded.size()
                  << ",\"load_ms\":" << load_ms
                  << ",\"load_current_engine_bytes\":" << load_allocation.current_bytes
                  << ",\"load_peak_engine_bytes\":" << load_allocation.peak_bytes
                  << ",\"load_host_to_device_calls\":"
                  << load_transfers.host_to_device_calls
                  << ",\"load_host_to_device_bytes\":"
                  << load_transfers.host_to_device_bytes
                  << ",\"load_device_to_host_calls\":"
                  << load_transfers.device_to_host_calls
                  << ",\"load_device_to_device_calls\":"
                  << load_transfers.device_to_device_calls
                  << ",\"parameter_count\":" << model.parameter_count()
                  << ",\"fp32_weight_bytes\":"
                  << external.model.weight_bytes(sizeof(float))
                  << ",\"bf16_training_mirror_tensors\":" << bf16_mirrors.size()
                  << ",\"bf16_training_mirror_bytes\":"
                  << [&] {
                         std::uint64_t bytes = 0;
                         for (const auto& [master, mirror] : bf16_mirrors) {
                             (void)master;
                             bytes += static_cast<std::uint64_t>(mirror->storage().num_bytes());
                         }
                         return bytes;
                     }()
                  << ",\"warmup\":" << warmup
                  << ",\"steps\":" << steps
                  << ",\"batch\":" << batch_size
                  << ",\"context\":" << input_ids.size()
                  << ",\"warmup_ms\":" << warmup_ms
                  << ",\"trained_tokens\":" << trained_tokens
                  << ",\"first_loss\":" << first_loss
                  << ",\"final_loss\":" << final_loss
                  << ",\"loss\":" << final_loss
                  << ",\"observed_parameter_before\":" << before
                  << ",\"observed_parameter_after\":" << after
                  << ",\"parameter_changed\":" << (before != after ? "true" : "false")
                  << ",\"step_ms\":" << measured_ms
                  << ",\"measured_ms\":" << measured_ms
                  << ",\"mean_step_ms\":" << measured_ms / static_cast<double>(steps)
                  << ",\"tokens_per_second\":"
                  << static_cast<double>(trained_tokens) * 1000.0 / measured_ms
                  << ",\"milliseconds_per_token\":"
                  << measured_ms / static_cast<double>(trained_tokens)
                  << ",\"optimizer_ms\":" << optimizer_ms
                  << ",\"mean_optimizer_ms\":" << optimizer_ms / static_cast<double>(steps)
                  << ",\"optimizer_timing_boundary\":\"post_backward_sync\""
                  << ",\"optimizer_host_to_device_calls\":" << optimizer_h2d
                  << ",\"optimizer_device_to_host_calls\":" << optimizer_d2h
                  << ",\"optimizer_host_to_device_bytes\":"
                  << optimizer_h2d_bytes
                  << ",\"optimizer_device_to_host_bytes\":"
                  << optimizer_d2h_bytes
                  << ",\"engine_current_bytes\":" << allocation.current_bytes
                  << ",\"engine_peak_bytes\":" << allocation.peak_bytes
                  << ",\"engine_total_allocated_bytes\":"
                  << allocation.total_allocated_bytes
                  << ",\"engine_allocation_calls\":" << allocation.allocation_calls
                  << ",\"engine_deallocation_calls\":" << allocation.deallocation_calls
                  << ",\"engine_backend_allocation_calls\":"
                  << allocation.backend_allocation_calls
                  << ",\"engine_backend_deallocation_calls\":"
                  << allocation.backend_deallocation_calls
                  << ",\"engine_cache_reuse_calls\":" << allocation.cache_reuse_calls
                  << ",\"engine_cached_bytes\":" << allocation.cached_bytes
                  << ",\"engine_reserved_bytes\":" << allocation.reserved_bytes
                  << "}\n";
        return before != after && std::isfinite(final_loss) ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "microllm_hf_train_step: " << error.what() << '\n';
        return 1;
    }
}
