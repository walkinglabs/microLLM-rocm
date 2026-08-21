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
    std::filesystem::path cache_logits_output;
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
    std::string prefill_logits_mode = "last";
    std::string batch_argmax_mode = "device";
    std::string decode_mode = "generation";
    std::string kv_cache_dtype = "fp32";
    std::string kv_cache_fp32_layers;
    std::int64_t cache_capacity = 0;
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
        else if (name == "--cache-logits-output") {
            result.cache_logits_output = argv[index + 1];
        }
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
        else if (name == "--prefill-logits") result.prefill_logits_mode = argv[index + 1];
        else if (name == "--batch-argmax-mode") result.batch_argmax_mode = argv[index + 1];
        else if (name == "--decode-mode") result.decode_mode = argv[index + 1];
        else if (name == "--kv-cache-dtype") result.kv_cache_dtype = argv[index + 1];
        else if (name == "--kv-cache-fp32-layers") {
            result.kv_cache_fp32_layers = argv[index + 1];
        }
        else if (name == "--cache-capacity") {
            result.cache_capacity = std::stoll(argv[index + 1]);
        }
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
    if (result.prefill_logits_mode != "last" && result.prefill_logits_mode != "full") {
        throw std::invalid_argument("--prefill-logits must be last or full");
    }
    if (result.batch_argmax_mode != "device" && result.batch_argmax_mode != "host") {
        throw std::invalid_argument("--batch-argmax-mode must be device or host");
    }
    if (result.decode_mode != "generation" && result.decode_mode != "steady") {
        throw std::invalid_argument("--decode-mode must be generation or steady");
    }
    if (result.kv_cache_dtype != "fp32" && result.kv_cache_dtype != "bf16") {
        throw std::invalid_argument("--kv-cache-dtype must be fp32 or bf16");
    }
    if (result.new_tokens < 0) throw std::invalid_argument("--new-tokens cannot be negative");
    if (result.cache_capacity < 0) {
        throw std::invalid_argument("--cache-capacity cannot be negative");
    }
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
    if (result.workload == "prefill" && result.new_tokens != 0) {
        throw std::invalid_argument("prefill workload requires --new-tokens 0");
    }
    if (result.bf16_attention && !result.bf16_ffn) {
        throw std::invalid_argument("--bf16-attention requires --bf16-ffn true");
    }
    if (!result.cache_logits_output.empty() &&
        (!result.use_cache || result.workload == "prefill" || result.new_tokens < 2)) {
        throw std::invalid_argument(
            "--cache-logits-output requires cached decode with at least two new tokens");
    }
    return result;
}

struct GenerationRun {
    std::vector<std::int32_t> suffix;
    std::size_t kv_cache_actual_bytes = 0;
    std::size_t kv_cache_active_bytes = 0;
    std::size_t kv_cache_element_bytes = 0;
    std::size_t kv_cache_fp32_bytes = 0;
    std::size_t kv_cache_bf16_bytes = 0;
    std::int64_t kv_cache_fp32_layers = 0;
    std::int64_t kv_cache_bf16_layers = 0;
    std::int64_t kv_cache_capacity_tokens = 0;
    std::int64_t kv_cache_active_tokens = 0;
};

std::size_t tensor_bytes(const microllm::Tensor& tensor) {
    return tensor.defined()
               ? static_cast<std::size_t>(tensor.numel()) * microllm::dtype_size(tensor.dtype())
               : 0;
}

struct CachedGenerationState {
    CachedGenerationState(std::vector<microllm::DType> layer_dtypes,
                          std::int64_t capacity, std::int64_t batch)
        : cache(std::move(layer_dtypes), capacity, batch) {}
    microllm::inference::KVCache cache;
    microllm::Tensor logits;
};

CachedGenerationState prepare_cached(
    microllm::model::TransformerModel& model,
    const std::vector<std::int32_t>& prompt, std::int64_t capacity,
    std::int64_t batch, bool full_prefill,
    const std::vector<microllm::DType>& cache_dtypes) {
    CachedGenerationState state(cache_dtypes, capacity, batch);
    std::vector<std::int32_t> batched_prompt;
    batched_prompt.reserve(prompt.size() * static_cast<std::size_t>(batch));
    for (std::int64_t row = 0; row < batch; ++row) {
        batched_prompt.insert(batched_prompt.end(), prompt.begin(), prompt.end());
    }
    if (full_prefill) {
        state.logits = model.forward_prefill_cached(
            microllm::Tensor::from_int32_vector(
                batched_prompt, {batch, static_cast<std::int64_t>(prompt.size())}),
            state.cache);
    } else {
        for (const auto token : prompt) {
            const std::vector<std::int32_t> row_tokens(
                static_cast<std::size_t>(batch), token);
            state.logits = model.forward_cached(
                microllm::Tensor::from_int32_vector(row_tokens, {batch, 1}),
                state.cache);
        }
    }
    return state;
}

GenerationRun decode_cached(microllm::model::TransformerModel& model,
                            CachedGenerationState& state,
                            std::int64_t new_tokens, bool steady = false) {
    GenerationRun run;
    run.kv_cache_capacity_tokens = state.cache.max_sequence_length();
    auto next_tensor = microllm::ops::argmax_last_dim(state.logits);
    for (std::int64_t generated = 0; generated < new_tokens; ++generated) {
        if (steady) {
            state.logits = model.forward_cached(next_tensor, state.cache);
            next_tensor = microllm::ops::argmax_last_dim(state.logits);
        }
        const auto next_rows = next_tensor.to_int32_vector();
        const auto next = next_rows.front();
        if (next < 0 || std::any_of(next_rows.begin(), next_rows.end(),
                                    [next](std::int32_t value) {
                                        return value != next;
                                    })) {
            throw std::runtime_error(
                "cached identical batch rows produced invalid or different tokens");
        }
        run.suffix.push_back(next);
        if (!steady && generated + 1 < new_tokens) {
            state.logits = model.forward_cached(next_tensor, state.cache);
            next_tensor = microllm::ops::argmax_last_dim(state.logits);
        }
    }
    run.kv_cache_active_tokens = state.cache.position();
    for (std::size_t layer = 0; layer < state.cache.layer_count(); ++layer) {
        const auto& layer_state = state.cache.layer(layer);
        if (state.cache.layer_dtype(layer) == microllm::DType::Float32) {
            ++run.kv_cache_fp32_layers;
        } else {
            ++run.kv_cache_bf16_layers;
        }
        for (const auto* tensor : {&layer_state.key, &layer_state.value}) {
            if (!tensor->defined()) continue;
            // The Tensor is an active-prefix view. Storage owns the full
            // preallocated capacity and is the allocation evidence.
            run.kv_cache_actual_bytes += tensor->storage().num_bytes();
            run.kv_cache_active_bytes += tensor_bytes(*tensor);
            if (tensor->dtype() == microllm::DType::Float32) {
                run.kv_cache_fp32_bytes += tensor->storage().num_bytes();
            } else {
                run.kv_cache_bf16_bytes += tensor->storage().num_bytes();
            }
        }
    }
    run.kv_cache_element_bytes = run.kv_cache_fp32_layers > 0 &&
                                         run.kv_cache_bf16_layers > 0
                                     ? 0U
                                     : run.kv_cache_fp32_layers > 0 ? 4U : 2U;
    return run;
}

using TokenRows = std::vector<std::vector<std::int32_t>>;

std::vector<std::int32_t> uncached_step(
    microllm::model::TransformerModel& model, TokenRows& sequences,
    bool device_argmax) {
    const auto batch = static_cast<std::int64_t>(sequences.size());
    const auto length = static_cast<std::int64_t>(sequences.front().size());
    std::vector<std::int32_t> flat;
    flat.reserve(static_cast<std::size_t>(batch * length));
    for (const auto& sequence : sequences) flat.insert(flat.end(), sequence.begin(), sequence.end());
    auto token_tensor = microllm::Tensor::from_int32_vector(flat, {batch, length});
    if (model.device().is_hip()) token_tensor = token_tensor.to(model.device());
    auto last_logits = model.forward_inference_last_logits(token_tensor)
                           .reshape({batch, model.config().vocabulary_size});
    std::vector<std::int32_t> selected;
    if (device_argmax) {
        selected = microllm::ops::argmax_last_dim(last_logits).to_int32_vector();
    } else {
        const auto host_logits = last_logits.to_vector();
        selected.resize(static_cast<std::size_t>(batch));
        const auto vocabulary = model.config().vocabulary_size;
        for (std::int64_t row = 0; row < batch; ++row) {
            const auto begin = host_logits.begin() + row * vocabulary;
            const auto maximum = std::max_element(begin, begin + vocabulary);
            if (!std::isfinite(*maximum)) {
                throw std::runtime_error("uncached generation produced non-finite logits");
            }
            selected[static_cast<std::size_t>(row)] =
                static_cast<std::int32_t>(std::distance(begin, maximum));
        }
    }
    for (std::int64_t row = 0; row < batch; ++row) {
        const auto next = selected[static_cast<std::size_t>(row)];
        if (next < 0) {
            throw std::runtime_error("uncached generation produced non-finite logits");
        }
        sequences[static_cast<std::size_t>(row)].push_back(next);
    }
    return selected;
}

GenerationRun decode_uncached(microllm::model::TransformerModel& model,
                              TokenRows& sequences, std::int64_t new_tokens,
                              bool device_argmax) {
    GenerationRun run;
    for (std::int64_t generated = 0; generated < new_tokens; ++generated) {
        const auto selected = uncached_step(model, sequences, device_argmax);
        run.suffix.push_back(selected.front());
    }
    for (std::int64_t row = 1; row < static_cast<std::int64_t>(sequences.size()); ++row) {
        if (sequences[static_cast<std::size_t>(row)] != sequences.front()) {
            throw std::runtime_error("identical batch rows generated different tokens");
        }
    }
    return run;
}

GenerationRun generate_uncached(microllm::model::TransformerModel& model,
                                const std::vector<std::int32_t>& prompt,
                                std::int64_t batch, std::int64_t new_tokens,
                                bool device_argmax) {
    std::vector<std::vector<std::int32_t>> sequences(
        static_cast<std::size_t>(batch), prompt);
    return decode_uncached(model, sequences, new_tokens, device_argmax);
}

std::vector<std::int32_t> nonnegative_values(std::string_view text,
                                             const char* error) {
    std::vector<std::int32_t> output;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto item = text.substr(0, comma);
        std::int32_t value = 0;
        const auto parsed = std::from_chars(item.data(), item.data() + item.size(), value);
        if (item.empty() || parsed.ec != std::errc{} || parsed.ptr != item.data() + item.size() ||
            value < 0) throw std::invalid_argument(error);
        output.push_back(value);
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    return output;
}

std::vector<std::int32_t> tokens(std::string_view text) {
    return nonnegative_values(
        text, "--tokens must be comma-separated nonnegative IDs");
}

std::vector<microllm::DType> cache_layer_dtypes(
    std::int64_t layers, microllm::DType base_dtype,
    std::string_view fp32_layers) {
    std::vector<microllm::DType> result(
        static_cast<std::size_t>(layers), base_dtype);
    if (fp32_layers.empty()) return result;
    std::vector<bool> seen(static_cast<std::size_t>(layers), false);
    for (const auto layer : nonnegative_values(
             fp32_layers,
             "--kv-cache-fp32-layers must be comma-separated nonnegative indices")) {
        if (layer >= layers || seen[static_cast<std::size_t>(layer)]) {
            throw std::invalid_argument(
                "--kv-cache-fp32-layers must contain unique in-range layer indices");
        }
        seen[static_cast<std::size_t>(layer)] = true;
        result[static_cast<std::size_t>(layer)] = microllm::DType::Float32;
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        auto external = microllm::model::load_huggingface_config(command.config);
        const auto cache_dtype = command.kv_cache_dtype == "bf16"
                                     ? microllm::DType::BFloat16
                                     : microllm::DType::Float32;
        const auto cache_dtypes = cache_layer_dtypes(
            external.model.layers, cache_dtype, command.kv_cache_fp32_layers);
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
            const auto prefill = [&]() {
                return command.prefill_logits_mode == "last"
                           ? model.forward_inference_last_logits(token_tensor)
                           : model.forward_inference(token_tensor);
            };
            for (int iteration = 0; iteration < command.prefill_warmup; ++iteration) {
                (void)prefill();
            }
            microllm::runtime::synchronize(device);
            if (device.is_hip()) microllm::runtime::enable_hip_caching_allocator(device);
            microllm::runtime::reset_allocation_peak(device);
            microllm::runtime::reset_transfer_stats();
            const auto forward_start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < command.prefill_steps; ++iteration) {
                logits_tensor = prefill();
            }
            microllm::runtime::synchronize(device);
            const auto forward_finish = std::chrono::steady_clock::now();
            forward_ms = std::chrono::duration<double, std::milli>(
                             forward_finish - forward_start).count();
        }
        const auto logits = run_prefill ? logits_tensor.to_vector() : std::vector<float>{};
        const auto vocabulary = static_cast<std::size_t>(external.model.vocabulary_size);
        const auto offset = command.prefill_logits_mode == "last"
                                ? 0U : (ids.size() - 1) * vocabulary;
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
        double decode_prepare_ms = 0.0;
        double warmup_ms = 0.0;
        std::vector<std::int32_t> generated_suffix;
        std::string generated_text;
        GenerationRun generation_evidence;
        microllm::Tensor cache_logits_evidence;
        if (run_decode && command.new_tokens > 0) {
            const auto minimum_cache_capacity =
                static_cast<std::int64_t>(ids.size()) + command.new_tokens;
            const auto cache_capacity = command.cache_capacity == 0
                                            ? minimum_cache_capacity
                                            : command.cache_capacity;
            if (command.use_cache && cache_capacity < minimum_cache_capacity) {
                throw std::invalid_argument(
                    "--cache-capacity is smaller than prompt plus decode steps");
            }
            const auto warmup_start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < command.warmup; ++iteration) {
                if (command.use_cache) {
                    auto state = prepare_cached(
                        model, ids, cache_capacity,
                        command.batch,
                        command.cache_prefill_mode == "full", cache_dtypes);
                    (void)decode_cached(model, state, command.new_tokens,
                                        command.decode_mode == "steady");
                } else {
                    if (command.decode_mode == "steady") {
                        TokenRows sequences(static_cast<std::size_t>(command.batch), ids);
                        (void)uncached_step(
                            model, sequences, command.batch_argmax_mode == "device");
                        (void)decode_uncached(
                            model, sequences, command.new_tokens,
                            command.batch_argmax_mode == "device");
                    } else {
                        (void)generate_uncached(
                            model, ids, command.batch, command.new_tokens,
                            command.batch_argmax_mode == "device");
                    }
                }
            }
            microllm::runtime::synchronize(device);
            const auto warmup_finish = std::chrono::steady_clock::now();
            warmup_ms = std::chrono::duration<double, std::milli>(warmup_finish - warmup_start)
                            .count();
            if (device.is_hip()) microllm::runtime::enable_hip_caching_allocator(device);
            microllm::runtime::reset_allocation_peak(device);
            microllm::runtime::reset_transfer_stats();
            for (int iteration = 0; iteration < command.steps; ++iteration) {
                GenerationRun current;
                if (command.use_cache) {
                    // Prompt ingestion establishes the cache but is a prefill
                    // phase, so it is deliberately outside decode timing.
                    const auto prepare_start = std::chrono::steady_clock::now();
                    auto state = prepare_cached(
                        model, ids, cache_capacity,
                        command.batch,
                        command.cache_prefill_mode == "full", cache_dtypes);
                    microllm::runtime::synchronize(device);
                    const auto prepare_finish = std::chrono::steady_clock::now();
                    decode_prepare_ms += std::chrono::duration<double, std::milli>(
                                            prepare_finish - prepare_start).count();
                    const auto decode_start = std::chrono::steady_clock::now();
                    current = decode_cached(model, state, command.new_tokens,
                                            command.decode_mode == "steady");
                    microllm::runtime::synchronize(device);
                    const auto decode_finish = std::chrono::steady_clock::now();
                    generation_ms += std::chrono::duration<double, std::milli>(
                                         decode_finish - decode_start).count();
                    if (iteration == 0 && !command.cache_logits_output.empty()) {
                        cache_logits_evidence = state.logits;
                    }
                } else {
                    TokenRows steady_sequences;
                    if (command.decode_mode == "steady") {
                        steady_sequences = TokenRows(
                            static_cast<std::size_t>(command.batch), ids);
                        const auto prepare_start = std::chrono::steady_clock::now();
                        (void)uncached_step(
                            model, steady_sequences,
                            command.batch_argmax_mode == "device");
                        microllm::runtime::synchronize(device);
                        const auto prepare_finish = std::chrono::steady_clock::now();
                        decode_prepare_ms += std::chrono::duration<double, std::milli>(
                                                 prepare_finish - prepare_start).count();
                    }
                    const auto decode_start = std::chrono::steady_clock::now();
                    current = command.decode_mode == "steady"
                                  ? decode_uncached(
                                        model, steady_sequences, command.new_tokens,
                                        command.batch_argmax_mode == "device")
                                  : generate_uncached(
                                        model, ids, command.batch, command.new_tokens,
                                        command.batch_argmax_mode == "device");
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
        const auto measured_transfers = microllm::runtime::transfer_stats();
        if (!command.cache_logits_output.empty()) {
            const auto cache_logits = cache_logits_evidence.to_vector();
            std::ofstream output(command.cache_logits_output,
                                 std::ios::binary | std::ios::trunc);
            if (!output) throw std::runtime_error("cannot open cached logits output");
            output.write(reinterpret_cast<const char*>(cache_logits.data()),
                         static_cast<std::streamsize>(cache_logits.size() * sizeof(float)));
            if (!output) throw std::runtime_error("failed writing cached logits output");
        }
        const auto info = device.is_cpu()
                              ? microllm::runtime::DeviceInfo{device, "host CPU", "host"}
                              : microllm::runtime::device_info(device);
        std::cout << std::setprecision(9)
                  << "{\"schema_version\":1,\"status\":\"pass\""
                  << ",\"device\":\"" << device.str() << "\""
                  << ",\"device_name\":\"" << info.name << "\""
                  << ",\"architecture\":\"" << info.architecture << "\""
                  << ",\"device_total_bytes\":" << info.total_memory
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
                  << ",\"requested_cache_capacity\":"
                  << command.cache_capacity
                  << ",\"prefill_logits_mode\":\""
                  << command.prefill_logits_mode << "\""
                  << ",\"kv_cache_dtype\":\""
                  << command.kv_cache_dtype << "\""
                  << ",\"kv_cache_fp32_layer_policy\":\""
                  << command.kv_cache_fp32_layers << "\""
                  << ",\"cache_logits_step\":"
                  << (command.cache_logits_output.empty()
                          ? 0 : command.new_tokens - 1)
                  << ",\"batch_argmax_mode\":\""
                  << command.batch_argmax_mode << "\""
                  << ",\"decode_mode\":\"" << command.decode_mode << "\""
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
                  << ",\"engine_peak_share_of_device\":"
                  << (info.total_memory == 0
                          ? 0.0
                          : static_cast<double>(allocation.peak_bytes) /
                                static_cast<double>(info.total_memory))
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
        std::cout << ",\"measured_h2d_calls\":"
                  << measured_transfers.host_to_device_calls
                  << ",\"measured_h2d_bytes\":"
                  << measured_transfers.host_to_device_bytes
                  << ",\"measured_d2h_calls\":"
                  << measured_transfers.device_to_host_calls
                  << ",\"measured_d2h_bytes\":"
                  << measured_transfers.device_to_host_bytes
                  << ",\"measured_d2d_calls\":"
                  << measured_transfers.device_to_device_calls
                  << ",\"measured_d2d_bytes\":"
                  << measured_transfers.device_to_device_bytes;
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
            const auto measured_forward_steps =
                command.decode_mode == "steady" || !command.use_cache
                    ? measured_tokens
                    : (generated_suffix.empty() ? 0U : generated_suffix.size() - 1U) *
                          static_cast<std::size_t>(command.steps) *
                          static_cast<std::size_t>(command.batch);
            std::cout << ",\"generation_ms\":" << generation_ms
                      << ",\"mean_generation_ms\":"
                      << generation_ms / static_cast<double>(command.steps)
                      << ",\"decode_prepare_ms\":" << decode_prepare_ms
                      << ",\"mean_decode_prepare_ms\":"
                      << decode_prepare_ms / static_cast<double>(command.steps)
                      << ",\"cache_prepare_ms\":"
                      << (command.use_cache ? decode_prepare_ms : 0.0)
                      << ",\"mean_cache_prepare_ms\":"
                      << (command.use_cache
                              ? decode_prepare_ms / static_cast<double>(command.steps)
                              : 0.0)
                      << ",\"mean_end_to_end_generation_ms\":"
                      << (decode_prepare_ms + generation_ms) /
                             static_cast<double>(command.steps)
                      << ",\"measured_tokens\":" << measured_tokens
                      << ",\"measured_forward_steps\":"
                      << measured_forward_steps
                      << ",\"decode_step_semantics\":\""
                      << (command.decode_mode == "steady" || !command.use_cache
                              ? "one_model_forward_per_measured_token"
                              : "generated_tokens_including_prefill_first")
                      << "\""
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
                      << ",\"kv_cache_fp32_layers\":"
                      << generation_evidence.kv_cache_fp32_layers
                      << ",\"kv_cache_bf16_layers\":"
                      << generation_evidence.kv_cache_bf16_layers
                      << ",\"kv_cache_fp32_bytes\":"
                      << generation_evidence.kv_cache_fp32_bytes
                      << ",\"kv_cache_bf16_bytes\":"
                      << generation_evidence.kv_cache_bf16_bytes
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
