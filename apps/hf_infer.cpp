#include <algorithm>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <microllm/model/huggingface.h>
#include <microllm/model/model.h>
#include <microllm/io/huggingface_bpe_tokenizer.h>
#include <microllm/io/chat_template.h>
#include <microllm/runtime/runtime.h>
#include <microllm/runtime/memory.h>
#include <microllm/inference/generator.h>
#include <microllm/inference/kv_cache.h>
#include <microllm/ops/ops.h>

namespace {

struct Options {
    std::filesystem::path config;
    std::filesystem::path weights;
    std::string tokens;
    std::string device = "cpu";
    std::int64_t top_k = 10;
    std::filesystem::path logits_output;
    std::string text;
    std::filesystem::path vocabulary;
    std::filesystem::path merges;
    std::int64_t new_tokens = 0;
    int warmup = 0;
    int steps = 1;
    int prefill_warmup = 0;
    int prefill_steps = 1;
    std::string tokenizer_family = "qwen2";
    std::string chat_user;
    bool bf16_ffn = false;
    bool bf16_attention = false;
    std::string workload = "both";
    std::int64_t batch = 1;
    bool use_cache = true;
    std::string cache_prefill_mode = "full";
};

Options options(int argc, char** argv) {
    Options result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::invalid_argument("missing CLI value");
        const std::string name = argv[index];
        if (name == "--config") result.config = argv[index + 1];
        else if (name == "--weights") result.weights = argv[index + 1];
        else if (name == "--tokens") result.tokens = argv[index + 1];
        else if (name == "--device") result.device = argv[index + 1];
        else if (name == "--top-k") result.top_k = std::stoll(argv[index + 1]);
        else if (name == "--logits-output") result.logits_output = argv[index + 1];
        else if (name == "--text") result.text = argv[index + 1];
        else if (name == "--vocab") result.vocabulary = argv[index + 1];
        else if (name == "--merges") result.merges = argv[index + 1];
        else if (name == "--new-tokens") result.new_tokens = std::stoll(argv[index + 1]);
        else if (name == "--warmup") result.warmup = std::stoi(argv[index + 1]);
        else if (name == "--steps") result.steps = std::stoi(argv[index + 1]);
        else if (name == "--prefill-warmup") {
            result.prefill_warmup = std::stoi(argv[index + 1]);
        } else if (name == "--prefill-steps") {
            result.prefill_steps = std::stoi(argv[index + 1]);
        }
        else if (name == "--tokenizer-family") result.tokenizer_family = argv[index + 1];
        else if (name == "--chat-user") result.chat_user = argv[index + 1];
        else if (name == "--bf16-ffn") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument("--bf16-ffn must be true or false");
            }
            result.bf16_ffn = value == "true";
        } else if (name == "--bf16-attention") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument("--bf16-attention must be true or false");
            }
            result.bf16_attention = value == "true";
        } else if (name == "--workload") result.workload = argv[index + 1];
        else if (name == "--batch") result.batch = std::stoll(argv[index + 1]);
        else if (name == "--use-cache") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument("--use-cache must be true or false");
            }
            result.use_cache = value == "true";
        }
        else if (name == "--cache-prefill-mode") result.cache_prefill_mode = argv[index + 1];
        else throw std::invalid_argument("unknown CLI option: " + name);
    }
    if (result.config.empty() || result.weights.empty()) {
        throw std::invalid_argument("--config and --weights are required");
    }
    const auto token_mode = !result.tokens.empty();
    const auto text_mode = (!result.text.empty() || !result.chat_user.empty()) &&
                           !result.vocabulary.empty() && !result.merges.empty();
    if (token_mode == text_mode) {
        throw std::invalid_argument(
            "provide either --tokens or all of --text/--vocab/--merges");
    }
    if (result.device != "cpu" && result.device != "hip") {
        throw std::invalid_argument("--device must be cpu or hip");
    }
    if (result.top_k <= 0) throw std::invalid_argument("--top-k must be positive");
    if (result.batch <= 0) throw std::invalid_argument("--batch must be positive");
    if (result.cache_prefill_mode != "full" && result.cache_prefill_mode != "token") {
        throw std::invalid_argument("--cache-prefill-mode must be full or token");
    }
    if (result.new_tokens < 0) throw std::invalid_argument("--new-tokens cannot be negative");
    if (result.warmup < 0 || result.steps <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --steps positive");
    }
    if (result.prefill_warmup < 0 || result.prefill_steps <= 0) {
        throw std::invalid_argument(
            "--prefill-warmup must be nonnegative and --prefill-steps positive");
    }
    if (result.workload != "both" && result.workload != "prefill" &&
        result.workload != "decode") {
        throw std::invalid_argument("--workload must be both, prefill, or decode");
    }
    if (result.workload == "decode" && result.new_tokens == 0) {
        throw std::invalid_argument("decode workload requires positive --new-tokens");
    }
    if (result.workload != "prefill" && result.use_cache && result.batch != 1) {
        throw std::invalid_argument(
            "cached decode currently supports batch 1; use --use-cache false for larger batches");
    }
    if (result.workload == "prefill" && result.new_tokens != 0) {
        throw std::invalid_argument("prefill workload requires --new-tokens 0");
    }
    if (result.bf16_attention && !result.bf16_ffn) {
        throw std::invalid_argument("--bf16-attention requires --bf16-ffn true");
    }
    return result;
}

struct GenerationRun {
    std::vector<std::int32_t> suffix;
    std::size_t kv_cache_actual_bytes = 0;
    std::size_t kv_cache_active_bytes = 0;
    std::size_t kv_cache_element_bytes = 0;
    std::int64_t kv_cache_capacity_tokens = 0;
    std::int64_t kv_cache_active_tokens = 0;
};

std::size_t tensor_bytes(const microllm::Tensor& tensor) {
    return tensor.defined()
               ? static_cast<std::size_t>(tensor.numel()) * microllm::dtype_size(tensor.dtype())
               : 0;
}

struct CachedGenerationState {
    CachedGenerationState(std::int64_t layers, std::int64_t capacity)
        : cache(layers, capacity) {}
    microllm::inference::KVCache cache;
    microllm::Tensor logits;
};

CachedGenerationState prepare_cached(
    microllm::model::TransformerModel& model,
    const std::vector<std::int32_t>& prompt, std::int64_t new_tokens,
    bool full_prefill) {
    CachedGenerationState state(
        model.config().layers,
        static_cast<std::int64_t>(prompt.size()) + new_tokens);
    if (full_prefill) {
        state.logits = model.forward_prefill_cached(
            microllm::Tensor::from_int32_vector(
                prompt, {1, static_cast<std::int64_t>(prompt.size())}),
            state.cache);
    } else {
        for (const auto token : prompt) {
            state.logits = model.forward_cached(
                microllm::Tensor::from_int32_vector({token}, {1, 1}), state.cache);
        }
    }
    return state;
}

GenerationRun decode_cached(microllm::model::TransformerModel& model,
                            CachedGenerationState& state,
                            std::int64_t new_tokens) {
    GenerationRun run;
    run.kv_cache_capacity_tokens = state.cache.max_sequence_length();
    for (std::int64_t generated = 0; generated < new_tokens; ++generated) {
        auto next_tensor = microllm::ops::argmax(state.logits);
        const auto next = next_tensor.to_int32_vector().front();
        if (next < 0) throw std::runtime_error("cached generation produced non-finite logits");
        run.suffix.push_back(next);
        if (generated + 1 < new_tokens) {
            state.logits = model.forward_cached(next_tensor, state.cache);
        }
    }
    run.kv_cache_active_tokens = state.cache.position();
    for (std::size_t layer = 0; layer < state.cache.layer_count(); ++layer) {
        const auto& layer_state = state.cache.layer(layer);
        for (const auto* tensor : {&layer_state.key, &layer_state.value}) {
            if (!tensor->defined()) continue;
            const auto element_bytes = microllm::dtype_size(tensor->dtype());
            if (run.kv_cache_element_bytes != 0 &&
                run.kv_cache_element_bytes != element_bytes) {
                throw std::runtime_error("KV cache tensors use mixed element sizes");
            }
            run.kv_cache_element_bytes = element_bytes;
            // The Tensor is an active-prefix view. Storage owns the full
            // preallocated capacity and is the allocation evidence.
            run.kv_cache_actual_bytes += tensor->storage().num_bytes();
            run.kv_cache_active_bytes += tensor_bytes(*tensor);
        }
    }
    return run;
}

GenerationRun generate_uncached(microllm::model::TransformerModel& model,
                                const std::vector<std::int32_t>& prompt,
                                std::int64_t batch, std::int64_t new_tokens) {
    std::vector<std::vector<std::int32_t>> sequences(
        static_cast<std::size_t>(batch), prompt);
    GenerationRun run;
    const auto vocabulary = model.config().vocabulary_size;
    for (std::int64_t generated = 0; generated < new_tokens; ++generated) {
        const auto length = static_cast<std::int64_t>(sequences.front().size());
        std::vector<std::int32_t> flat;
        flat.reserve(static_cast<std::size_t>(batch * length));
        for (const auto& sequence : sequences) flat.insert(flat.end(), sequence.begin(), sequence.end());
        auto token_tensor = microllm::Tensor::from_int32_vector(flat, {batch, length});
        if (model.device().is_hip()) token_tensor = token_tensor.to(model.device());
        auto last_logits = model.forward_inference(token_tensor)
                               .slice(1, length - 1, length)
                               .contiguous()
                               .to_vector();
        for (std::int64_t row = 0; row < batch; ++row) {
            const auto begin = last_logits.begin() + row * vocabulary;
            const auto end = begin + vocabulary;
            const auto maximum = std::max_element(begin, end);
            if (maximum == end || !std::isfinite(*maximum)) {
                throw std::runtime_error("uncached generation produced non-finite logits");
            }
            sequences[static_cast<std::size_t>(row)].push_back(
                static_cast<std::int32_t>(std::distance(begin, maximum)));
        }
    }
    for (std::int64_t row = 1; row < batch; ++row) {
        if (sequences[static_cast<std::size_t>(row)] != sequences.front()) {
            throw std::runtime_error("identical batch rows generated different tokens");
        }
    }
    run.suffix.assign(sequences.front().begin() + static_cast<std::ptrdiff_t>(prompt.size()),
                      sequences.front().end());
    return run;
}

std::vector<std::int32_t> tokens(std::string_view text) {
    std::vector<std::int32_t> output;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto item = text.substr(0, comma);
        std::int32_t value = 0;
        const auto parsed = std::from_chars(item.data(), item.data() + item.size(), value);
        if (item.empty() || parsed.ec != std::errc{} || parsed.ptr != item.data() + item.size() ||
            value < 0) throw std::invalid_argument("--tokens must be comma-separated nonnegative IDs");
        output.push_back(value);
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    return output;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        auto external = microllm::model::load_huggingface_config(command.config);
        const auto device = command.device == "hip" ? microllm::Device::hip(0)
                                                     : microllm::Device::cpu();
        if (device.is_hip() && microllm::runtime::hip_device_count() == 0) {
            throw std::runtime_error("HIP inference requested without a visible device");
        }
        microllm::runtime::reset_allocation_peak(device);
        microllm::model::TransformerModel model(
            external.model, 1,
            microllm::model::ParameterInitialization::Uninitialized);
        model.to(device);
        microllm::model::LoadWeightsOptions load_options;
        load_options.mapping = microllm::model::qwen_style_weight_mapping(external.model);
        const auto load_start = std::chrono::steady_clock::now();
        const auto report = model.load_safetensors(command.weights, load_options);
        const auto load_finish = std::chrono::steady_clock::now();
        microllm::model::Bf16FfnPreparationReport bf16_report;
        microllm::model::Bf16WeightPreparationReport bf16_attention_report;
        microllm::runtime::reset_allocation_peak(device);
        const auto preparation_start = std::chrono::steady_clock::now();
        if (command.bf16_ffn) bf16_report = model.prepare_bf16_ffn_inference();
        if (command.bf16_attention) {
            bf16_attention_report = model.prepare_bf16_attention_inference();
        }
        microllm::runtime::synchronize(device);
        const auto preparation_finish = std::chrono::steady_clock::now();
        const auto preparation_allocation = microllm::runtime::allocation_stats(device);
        std::optional<microllm::io::HuggingFaceBpeTokenizer> tokenizer;
        std::vector<std::int32_t> ids;
        if (!command.tokens.empty()) {
            ids = tokens(command.tokens);
        } else {
            tokenizer = microllm::io::HuggingFaceBpeTokenizer::load(
                command.vocabulary, command.merges);
            if (command.tokenizer_family == "deepseek-distill") {
                tokenizer->add_special_token("<｜end▁of▁sentence｜>", 151643);
                tokenizer->add_special_token("<｜User｜>", 151644);
                tokenizer->add_special_token("<｜Assistant｜>", 151645);
                tokenizer->add_special_token("<｜begin▁of▁sentence｜>", 151646);
                tokenizer->add_special_token("<think>", 151648);
                tokenizer->add_special_token("</think>", 151649);
            } else if (command.tokenizer_family == "qwen2") {
                tokenizer->add_special_token("<|endoftext|>", 151643);
                tokenizer->add_special_token("<|im_start|>", 151644);
                tokenizer->add_special_token("<|im_end|>", 151645);
            } else {
                throw std::invalid_argument("unknown tokenizer family");
            }
            const auto prompt = command.chat_user.empty()
                                    ? command.text
                                    : command.tokenizer_family == "deepseek-distill"
                                          ? microllm::io::render_deepseek_distill_chat(
                                                {{"user", command.chat_user}})
                                          : microllm::io::render_qwen2_chat(
                                                {{"user", command.chat_user}});
            ids = tokenizer->encode(prompt);
        }
        if (ids.size() > static_cast<std::size_t>(external.model.max_sequence_length)) {
            throw std::invalid_argument("token sequence exceeds model context");
        }
        if (ids.empty()) throw std::invalid_argument("token sequence cannot be empty");
        std::vector<std::int32_t> batched_ids;
        batched_ids.reserve(ids.size() * static_cast<std::size_t>(command.batch));
        for (std::int64_t row = 0; row < command.batch; ++row) {
            batched_ids.insert(batched_ids.end(), ids.begin(), ids.end());
        }
        auto token_tensor = microllm::Tensor::from_int32_vector(
            batched_ids, {command.batch, static_cast<std::int64_t>(ids.size())});
        if (device.is_hip()) token_tensor = token_tensor.to(device);
        microllm::Tensor logits_tensor;
        double forward_ms = 0.0;
        const auto run_prefill = command.workload != "decode";
        const auto run_decode = command.workload != "prefill";
        if (run_prefill) {
            for (int iteration = 0; iteration < command.prefill_warmup; ++iteration) {
                (void)model.forward_inference(token_tensor);
            }
            microllm::runtime::synchronize(device);
            if (device.is_hip()) microllm::runtime::enable_hip_caching_allocator(device);
            microllm::runtime::reset_allocation_peak(device);
            const auto forward_start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < command.prefill_steps; ++iteration) {
                logits_tensor = model.forward_inference(token_tensor);
            }
            microllm::runtime::synchronize(device);
            const auto forward_finish = std::chrono::steady_clock::now();
            forward_ms = std::chrono::duration<double, std::milli>(
                             forward_finish - forward_start).count();
        }
        const auto logits = run_prefill ? logits_tensor.to_vector() : std::vector<float>{};
        const auto vocabulary = static_cast<std::size_t>(external.model.vocabulary_size);
        const auto offset = (ids.size() - 1) * vocabulary;
        std::vector<std::size_t> order(vocabulary);
        std::iota(order.begin(), order.end(), 0U);
        const auto selected = std::min<std::size_t>(static_cast<std::size_t>(command.top_k),
                                                    vocabulary);
        if (!command.logits_output.empty() && !run_prefill) {
            throw std::invalid_argument("--logits-output requires prefill or both workload");
        }
        if (!command.logits_output.empty()) {
            std::ofstream output(command.logits_output, std::ios::binary | std::ios::trunc);
            if (!output) throw std::runtime_error("cannot open logits output");
            output.write(reinterpret_cast<const char*>(logits.data() + offset),
                         static_cast<std::streamsize>(vocabulary * sizeof(float)));
            if (!output) throw std::runtime_error("failed writing logits output");
        }
        if (run_prefill) {
            std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(selected),
                              order.end(), [&](std::size_t left, std::size_t right) {
                                  return logits[offset + left] > logits[offset + right];
                              });
        }
        double generation_ms = 0.0;
        double cache_prepare_ms = 0.0;
        double warmup_ms = 0.0;
        std::vector<std::int32_t> generated_suffix;
        std::string generated_text;
        GenerationRun generation_evidence;
        if (run_decode && command.new_tokens > 0) {
            const auto warmup_start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < command.warmup; ++iteration) {
                if (command.use_cache) {
                    auto state = prepare_cached(
                        model, ids, command.new_tokens,
                        command.cache_prefill_mode == "full");
                    (void)decode_cached(model, state, command.new_tokens);
                } else {
                    (void)generate_uncached(
                        model, ids, command.batch, command.new_tokens);
                }
            }
            microllm::runtime::synchronize(device);
            const auto warmup_finish = std::chrono::steady_clock::now();
            warmup_ms = std::chrono::duration<double, std::milli>(warmup_finish - warmup_start)
                            .count();
            if (device.is_hip()) microllm::runtime::enable_hip_caching_allocator(device);
            microllm::runtime::reset_allocation_peak(device);
            for (int iteration = 0; iteration < command.steps; ++iteration) {
                GenerationRun current;
                if (command.use_cache) {
                    // Prompt ingestion establishes the cache but is a prefill
                    // phase, so it is deliberately outside decode timing.
                    const auto prepare_start = std::chrono::steady_clock::now();
                    auto state = prepare_cached(
                        model, ids, command.new_tokens,
                        command.cache_prefill_mode == "full");
                    microllm::runtime::synchronize(device);
                    const auto prepare_finish = std::chrono::steady_clock::now();
                    cache_prepare_ms += std::chrono::duration<double, std::milli>(
                                            prepare_finish - prepare_start).count();
                    const auto decode_start = std::chrono::steady_clock::now();
                    current = decode_cached(model, state, command.new_tokens);
                    microllm::runtime::synchronize(device);
                    const auto decode_finish = std::chrono::steady_clock::now();
                    generation_ms += std::chrono::duration<double, std::milli>(
                                         decode_finish - decode_start).count();
                } else {
                    const auto decode_start = std::chrono::steady_clock::now();
                    current = generate_uncached(
                        model, ids, command.batch, command.new_tokens);
                    microllm::runtime::synchronize(device);
                    const auto decode_finish = std::chrono::steady_clock::now();
                    generation_ms += std::chrono::duration<double, std::milli>(
                                         decode_finish - decode_start).count();
                }
                if (iteration != 0 && current.suffix != generated_suffix) {
                    throw std::runtime_error("deterministic generation changed across steps");
                }
                generated_suffix = current.suffix;
                generation_evidence = current;
            }
            if (tokenizer.has_value()) generated_text = tokenizer->decode(generated_suffix);
        }
        const auto allocation = microllm::runtime::allocation_stats(device);
        const auto info = device.is_cpu()
                              ? microllm::runtime::DeviceInfo{device, "host CPU", "host"}
                              : microllm::runtime::device_info(device);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"device\":\"" << device.str() << "\""
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"hip_runtime_version\":"
                  << microllm::runtime::hip_runtime_version()
                  << ",\"hip_driver_version\":" << microllm::runtime::hip_driver_version()
                  << ",\"compute_dtype\":\""
                  << (command.bf16_attention
                          ? "float32_with_bf16_ffn_attention"
                          : command.bf16_ffn ? "float32_with_bf16_ffn" : "float32")
                  << "\""
                  << ",\"inference_weight_policy\":\""
                  << (command.bf16_attention
                          ? "single_representation_bf16_ffn_attention"
                          : command.bf16_ffn ? "single_representation_bf16_ffn" : "float32")
                  << "\""
                  << ",\"workload\":\"" << command.workload << "\""
                  << ",\"bf16_ffn_converted_tensors\":"
                  << bf16_report.converted_tensors
                  << ",\"bf16_attention_converted_tensors\":"
                  << bf16_attention_report.converted_tensors
                  << ",\"fp32_weight_bytes_released\":"
                  << bf16_report.fp32_bytes_released
                  + bf16_attention_report.fp32_bytes_released
                  << ",\"bf16_weight_bytes_retained\":"
                  << bf16_report.bf16_bytes_retained
                  + bf16_attention_report.bf16_bytes_retained
                  << ",\"resident_weight_bytes\":"
                  << external.model.weight_bytes(sizeof(float)) -
                         bf16_report.fp32_bytes_released +
                         bf16_report.bf16_bytes_retained -
                         bf16_attention_report.fp32_bytes_released +
                         bf16_attention_report.bf16_bytes_retained
                  << ",\"measurement_profile\":\""
                  << (command.warmup > 0 || command.steps > 1 ||
                              command.prefill_warmup > 0 || command.prefill_steps > 1
                          ? "comparison" : "smoke")
                  << "\""
                  << ",\"parameter_count\":" << model.parameter_count()
                  << ",\"fp32_weight_bytes\":"
                  << external.model.weight_bytes(sizeof(float))
                  << ",\"loaded_tensors\":" << report.loaded.size()
                  << ",\"token_count\":" << ids.size()
                  << ",\"batch\":" << command.batch
                  << ",\"use_cache\":" << (command.use_cache ? "true" : "false")
                  << ",\"cache_prefill_mode\":\""
                  << command.cache_prefill_mode << "\""
                  << ",\"warmup\":" << command.warmup
                  << ",\"steps\":" << command.steps
                  << ",\"warmup_ms\":" << warmup_ms
                  << ",\"load_ms\":"
                  << std::chrono::duration<double, std::milli>(load_finish - load_start).count()
                  << ",\"bf16_prepare_ms\":"
                  << std::chrono::duration<double, std::milli>(
                         preparation_finish - preparation_start).count()
                  << ",\"preparation_current_bytes\":"
                  << preparation_allocation.current_bytes
                  << ",\"preparation_peak_bytes\":"
                  << preparation_allocation.peak_bytes
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
                  << ",\"engine_reserved_bytes\":" << allocation.reserved_bytes;
        if (run_prefill) {
            std::cout << ",\"prefill_warmup\":" << command.prefill_warmup
                      << ",\"prefill_steps\":" << command.prefill_steps
                      << ",\"forward_ms\":"
                      << forward_ms / static_cast<double>(command.prefill_steps)
                      << ",\"prefill_tokens_per_second\":"
                      << static_cast<double>(ids.size()) *
                             static_cast<double>(command.batch) *
                             command.prefill_steps * 1000.0 / forward_ms
                      << ",\"top_logits\":[";
            for (std::size_t index = 0; index < selected; ++index) {
                if (index != 0) std::cout << ',';
                std::cout << "{\"token\":" << order[index]
                          << ",\"logit\":" << logits[offset + order[index]] << '}';
            }
            std::cout << ']';
        }
        if (!generated_suffix.empty()) {
            const auto measured_tokens =
                generated_suffix.size() * static_cast<std::size_t>(command.steps) *
                static_cast<std::size_t>(command.batch);
            std::cout << ",\"generation_ms\":" << generation_ms
                      << ",\"mean_generation_ms\":"
                      << generation_ms / static_cast<double>(command.steps)
                      << ",\"cache_prepare_ms\":" << cache_prepare_ms
                      << ",\"mean_cache_prepare_ms\":"
                      << cache_prepare_ms / static_cast<double>(command.steps)
                      << ",\"mean_end_to_end_generation_ms\":"
                      << (cache_prepare_ms + generation_ms) /
                             static_cast<double>(command.steps)
                      << ",\"measured_tokens\":" << measured_tokens
                      << ",\"decode_tokens_per_second\":"
                      << static_cast<double>(measured_tokens) * 1000.0 / generation_ms
                      << ",\"decode_milliseconds_per_token\":"
                      << generation_ms / static_cast<double>(measured_tokens)
                      << ",\"kv_cache_actual_bytes\":"
                      << generation_evidence.kv_cache_actual_bytes
                      << ",\"kv_cache_active_bytes\":"
                      << generation_evidence.kv_cache_active_bytes
                      << ",\"kv_cache_capacity_tokens\":"
                      << generation_evidence.kv_cache_capacity_tokens
                      << ",\"kv_cache_active_tokens\":"
                      << generation_evidence.kv_cache_active_tokens
                      << ",\"kv_cache_layers\":" << external.model.layers
                      << ",\"kv_cache_heads\":" << external.model.kv_heads
                      << ",\"kv_cache_head_dimension\":"
                      << external.model.head_dimension()
                      << ",\"kv_cache_element_bytes\":"
                      << generation_evidence.kv_cache_element_bytes
                      << ",\"kv_cache_utilization\":"
                      << (generation_evidence.kv_cache_actual_bytes == 0
                              ? 0.0
                              : static_cast<double>(generation_evidence.kv_cache_active_bytes) /
                                    static_cast<double>(generation_evidence.kv_cache_actual_bytes));
            std::cout << ",\"generated_tokens\":[";
            for (std::size_t index = 0; index < generated_suffix.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << generated_suffix[index];
            }
            std::cout << ']';
            if (tokenizer.has_value()) {
                std::cout << ",\"generated_text\":\"";
                for (const auto character : generated_text) {
                    if (character == '"' || character == '\\') std::cout << '\\';
                    if (character == '\n') std::cout << "\\n";
                    else if (character == '\r') std::cout << "\\r";
                    else if (character == '\t') std::cout << "\\t";
                    else std::cout << character;
                }
                std::cout << '"';
            }
        }
        std::cout << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "microllm_hf_infer: " << error.what() << '\n';
        return 1;
    }
}
