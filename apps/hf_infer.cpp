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
#include <microllm/runtime/diagnostics.h>
#include <microllm/inference/generator.h>
#include <microllm/inference/kv_cache.h>
#include <microllm/inference/scheduler.h>
#include <microllm/ops/ops.h>
#include <microllm/profiling/trace.h>

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
    bool bf16_ffn_arena = false;
    std::int64_t bf16_ffn_arena_minimum_rows = 1;
    bool bf16_ffn_norm_fusion = false;
    bool bf16_ffn_norm_fusion_explicit = false;
    bool bf16_qkv_arena = false;
    std::int64_t bf16_qkv_arena_minimum_rows = 512;
    bool bf16_attention_norm_fusion = false;
    bool bf16_attention_norm_fusion_explicit = false;
    bool attention_core_arena = false;
    std::int64_t attention_core_arena_minimum_sequence = 512;
    std::int64_t cached_attention_splits = 0;
    std::int64_t cached_attention_minimum_sequence = 512;
    bool bf16_attention = false;
    bool fp8_linear = false;
    float fp8_activation_scale = 0.025F;
    float fp8_activation_minimum_scale = 1.0e-4F;
    float fp8_weight_scale = 0.005F;
    std::string fp8_weight_scale_mode = "fixed";
    std::string fp8_weight_scale_scope = "all-linear";
    std::string fp8_activation_scale_mode = "fixed";
    std::string fp8_diagnostic_mode = "full";
    std::string fp8_fp32_layers;
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
    std::int64_t continuous_slots = 0;
    std::string continuous_cache_buckets;
    std::string continuous_prompt_lengths;
    std::string continuous_new_token_lengths;
    std::string continuous_prompt_offsets;
    std::string continuous_arrival_steps;
    bool continuous_prefill_batch = true;
    bool continuous_diagnostics = false;
    bool continuous_bucket_overflow = false;
    std::filesystem::path trace_output;
    std::int64_t trace_max_elements = 4096;
    bool trace_all_layer_details = false;
    std::string trace_value_filter;
    int bf16_algorithm_index = -1;
    int bf16_grouped_qkv_algorithm_index = -1;
    int bf16_grouped_gate_up_algorithm_index = -1;
    bool bf16_grouped_gate_up_swish = false;
    bool bf16_grouped_qkv_prewarm = false;
    int fp32_attention_qk_solution_index = -1;
    int fp32_attention_pv_solution_index = -1;
    bool allocation_source_diagnostics = false;
    bool strided_copy_diagnostics = false;
    bool inference_bthd_attention = false;
    bool inference_bthd_bf16_qk = false;
    bool inference_bthd_online_attention = false;
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
        } else if (name == "--bf16-ffn-arena") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--bf16-ffn-arena must be true or false");
            }
            result.bf16_ffn_arena = value == "true";
        } else if (name == "--bf16-ffn-arena-minimum-rows") {
            result.bf16_ffn_arena_minimum_rows =
                std::stoll(argv[index + 1]);
        } else if (name == "--bf16-ffn-norm-fusion") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--bf16-ffn-norm-fusion must be true or false");
            }
            result.bf16_ffn_norm_fusion = value == "true";
            result.bf16_ffn_norm_fusion_explicit = true;
        } else if (name == "--bf16-qkv-arena") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--bf16-qkv-arena must be true or false");
            }
            result.bf16_qkv_arena = value == "true";
        } else if (name == "--bf16-qkv-arena-minimum-rows") {
            result.bf16_qkv_arena_minimum_rows =
                std::stoll(argv[index + 1]);
        } else if (name == "--bf16-attention-norm-fusion") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--bf16-attention-norm-fusion must be true or false");
            }
            result.bf16_attention_norm_fusion = value == "true";
            result.bf16_attention_norm_fusion_explicit = true;
        } else if (name == "--attention-core-arena") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--attention-core-arena must be true or false");
            }
            result.attention_core_arena = value == "true";
        } else if (name == "--attention-core-arena-minimum-sequence") {
            result.attention_core_arena_minimum_sequence =
                std::stoll(argv[index + 1]);
        } else if (name == "--cached-attention-splits") {
            result.cached_attention_splits = std::stoll(argv[index + 1]);
        } else if (name == "--cached-attention-minimum-sequence") {
            result.cached_attention_minimum_sequence =
                std::stoll(argv[index + 1]);
        } else if (name == "--bf16-attention") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument("--bf16-attention must be true or false");
            }
            result.bf16_attention = value == "true";
        } else if (name == "--workload") result.workload = argv[index + 1];
        else if (name == "--fp8-linear") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument("--fp8-linear must be true or false");
            }
            result.fp8_linear = value == "true";
        }
        else if (name == "--fp8-activation-scale") {
            result.fp8_activation_scale = std::stof(argv[index + 1]);
        }
        else if (name == "--fp8-activation-minimum-scale") {
            result.fp8_activation_minimum_scale = std::stof(argv[index + 1]);
        }
        else if (name == "--fp8-weight-scale") {
            result.fp8_weight_scale = std::stof(argv[index + 1]);
        }
        else if (name == "--fp8-weight-scale-mode") {
            result.fp8_weight_scale_mode = argv[index + 1];
        }
        else if (name == "--fp8-weight-scale-scope") {
            result.fp8_weight_scale_scope = argv[index + 1];
        }
        else if (name == "--fp8-activation-scale-mode") {
            result.fp8_activation_scale_mode = argv[index + 1];
        }
        else if (name == "--fp8-diagnostic-mode") {
            result.fp8_diagnostic_mode = argv[index + 1];
        }
        else if (name == "--fp8-fp32-layers") {
            result.fp8_fp32_layers = argv[index + 1];
        }
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
        else if (name == "--continuous-slots") {
            result.continuous_slots = std::stoll(argv[index + 1]);
        }
        else if (name == "--continuous-cache-buckets") {
            result.continuous_cache_buckets = argv[index + 1];
        }
        else if (name == "--continuous-prompt-lengths") {
            result.continuous_prompt_lengths = argv[index + 1];
        }
        else if (name == "--continuous-new-token-lengths") {
            result.continuous_new_token_lengths = argv[index + 1];
        }
        else if (name == "--continuous-prompt-offsets") {
            result.continuous_prompt_offsets = argv[index + 1];
        }
        else if (name == "--continuous-arrival-steps") {
            result.continuous_arrival_steps = argv[index + 1];
        }
        else if (name == "--continuous-diagnostics") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--continuous-diagnostics must be true or false");
            }
            result.continuous_diagnostics = value == "true";
        }
        else if (name == "--continuous-prefill-batch") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--continuous-prefill-batch must be true or false");
            }
            result.continuous_prefill_batch = value == "true";
        }
        else if (name == "--continuous-bucket-overflow") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--continuous-bucket-overflow must be true or false");
            }
            result.continuous_bucket_overflow = value == "true";
        }
        else if (name == "--trace-output") {
            result.trace_output = argv[index + 1];
        }
        else if (name == "--trace-max-elements") {
            result.trace_max_elements = std::stoll(argv[index + 1]);
        }
        else if (name == "--trace-all-layer-details") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--trace-all-layer-details must be true or false");
            }
            result.trace_all_layer_details = value == "true";
        }
        else if (name == "--trace-value-filter") {
            result.trace_value_filter = argv[index + 1];
        }
        else if (name == "--bf16-algorithm-index") {
            result.bf16_algorithm_index = std::stoi(argv[index + 1]);
        }
        else if (name == "--bf16-grouped-qkv-algorithm-index") {
            result.bf16_grouped_qkv_algorithm_index =
                std::stoi(argv[index + 1]);
        }
        else if (name == "--bf16-grouped-gate-up-algorithm-index") {
            result.bf16_grouped_gate_up_algorithm_index =
                std::stoi(argv[index + 1]);
        }
        else if (name == "--bf16-grouped-gate-up-swish") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--bf16-grouped-gate-up-swish must be true or false");
            }
            result.bf16_grouped_gate_up_swish = value == "true";
        }
        else if (name == "--bf16-grouped-qkv-prewarm") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--bf16-grouped-qkv-prewarm must be true or false");
            }
            result.bf16_grouped_qkv_prewarm = value == "true";
        }
        else if (name == "--fp32-attention-qk-solution-index") {
            result.fp32_attention_qk_solution_index =
                std::stoi(argv[index + 1]);
        }
        else if (name == "--fp32-attention-pv-solution-index") {
            result.fp32_attention_pv_solution_index =
                std::stoi(argv[index + 1]);
        }
        else if (name == "--allocation-source-diagnostics") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--allocation-source-diagnostics must be true or false");
            }
            result.allocation_source_diagnostics = value == "true";
        }
        else if (name == "--strided-copy-diagnostics") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--strided-copy-diagnostics must be true or false");
            }
            result.strided_copy_diagnostics = value == "true";
        }
        else if (name == "--inference-bthd-attention") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--inference-bthd-attention must be true or false");
            }
            result.inference_bthd_attention = value == "true";
        }
        else if (name == "--inference-bthd-bf16-qk") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--inference-bthd-bf16-qk must be true or false");
            }
            result.inference_bthd_bf16_qk = value == "true";
        }
        else if (name == "--inference-bthd-online-attention") {
            const std::string value = argv[index + 1];
            if (value != "true" && value != "false") {
                throw std::invalid_argument(
                    "--inference-bthd-online-attention must be true or false");
            }
            result.inference_bthd_online_attention = value == "true";
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
        result.workload != "decode" && result.workload != "continuous") {
        throw std::invalid_argument(
            "--workload must be both, prefill, decode, or continuous");
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
    if (result.bf16_ffn_arena && !result.bf16_ffn) {
        throw std::invalid_argument(
            "--bf16-ffn-arena requires --bf16-ffn true");
    }
    if (!result.bf16_ffn_norm_fusion_explicit && result.bf16_ffn_arena) {
        result.bf16_ffn_norm_fusion = true;
    }
    if (result.bf16_ffn_norm_fusion && !result.bf16_ffn_arena) {
        throw std::invalid_argument(
            "--bf16-ffn-norm-fusion requires BF16 FFN Arena");
    }
    if (result.bf16_ffn_arena_minimum_rows <= 0 ||
        (!result.bf16_ffn_arena &&
         result.bf16_ffn_arena_minimum_rows != 1)) {
        throw std::invalid_argument(
            "--bf16-ffn-arena-minimum-rows must be positive and requires Arena");
    }
    if (result.bf16_qkv_arena && !result.bf16_attention) {
        throw std::invalid_argument(
            "--bf16-qkv-arena requires --bf16-attention true");
    }
    if (result.bf16_qkv_arena_minimum_rows <= 0 ||
        (!result.bf16_qkv_arena &&
         result.bf16_qkv_arena_minimum_rows != 512)) {
        throw std::invalid_argument(
            "--bf16-qkv-arena-minimum-rows must be positive and requires QKV Arena");
    }
    if (!result.bf16_attention_norm_fusion_explicit && result.bf16_qkv_arena) {
        result.bf16_attention_norm_fusion = true;
    }
    if (result.bf16_attention_norm_fusion && !result.bf16_qkv_arena) {
        throw std::invalid_argument(
            "--bf16-attention-norm-fusion requires BF16 QKV Arena");
    }
    if (result.attention_core_arena_minimum_sequence <= 0 ||
        (!result.attention_core_arena &&
         result.attention_core_arena_minimum_sequence != 512)) {
        throw std::invalid_argument(
            "--attention-core-arena-minimum-sequence must be positive and requires core Arena");
    }
    if (result.cached_attention_splits < 0 ||
        result.cached_attention_splits > 32 ||
        result.cached_attention_minimum_sequence <= 0) {
        throw std::invalid_argument(
            "--cached-attention-splits must be 0..32 and minimum sequence positive");
    }
    if (result.bf16_ffn_arena && !result.trace_output.empty()) {
        throw std::invalid_argument(
            "--bf16-ffn-arena is unavailable during value tracing");
    }
    if (result.bf16_qkv_arena && !result.trace_output.empty()) {
        throw std::invalid_argument(
            "--bf16-qkv-arena is unavailable during value tracing");
    }
    if ((result.fp8_linear && (result.bf16_ffn || result.bf16_attention)) ||
        !std::isfinite(result.fp8_activation_scale) ||
        result.fp8_activation_scale <= 0.0F ||
        !std::isfinite(result.fp8_activation_minimum_scale) ||
        result.fp8_activation_minimum_scale <= 0.0F ||
        !std::isfinite(result.fp8_weight_scale) ||
        result.fp8_weight_scale <= 0.0F) {
        throw std::invalid_argument(
            "FP8 Linear requires positive finite scales and is exclusive with BF16 preparation");
    }
    if (result.fp8_weight_scale_mode != "fixed" &&
        result.fp8_weight_scale_mode != "tensor-amax" &&
        result.fp8_weight_scale_mode != "device-tensor-amax" &&
        result.fp8_weight_scale_mode != "output-channel-amax") {
        throw std::invalid_argument(
            "--fp8-weight-scale-mode must be fixed, tensor-amax, device-tensor-amax, or output-channel-amax");
    }
    if (result.fp8_activation_scale_mode != "fixed" &&
        result.fp8_activation_scale_mode != "tensor-amax" &&
        result.fp8_activation_scale_mode != "ffn-outer-row") {
        throw std::invalid_argument(
            "--fp8-activation-scale-mode must be fixed, tensor-amax, or ffn-outer-row");
    }
    if (result.fp8_diagnostic_mode != "full" &&
        result.fp8_diagnostic_mode != "weight-only" &&
        result.fp8_diagnostic_mode != "activation-only" &&
        result.fp8_diagnostic_mode != "both-roundtrip") {
        throw std::invalid_argument(
            "--fp8-diagnostic-mode must be full, weight-only, activation-only, or both-roundtrip");
    }
    if (result.fp8_weight_scale_scope != "all-linear" &&
        result.fp8_weight_scale_scope != "attention-output-only") {
        throw std::invalid_argument(
            "--fp8-weight-scale-scope must be all-linear or attention-output-only");
    }
    if (result.fp8_weight_scale_scope != "all-linear" &&
        result.fp8_weight_scale_mode != "output-channel-amax") {
        throw std::invalid_argument(
            "--fp8-weight-scale-scope requires output-channel-amax weights");
    }
    if (result.fp8_diagnostic_mode != "full" && !result.fp8_linear) {
        throw std::invalid_argument(
            "--fp8-diagnostic-mode requires --fp8-linear true");
    }
    if (!result.fp8_fp32_layers.empty() && !result.fp8_linear) {
        throw std::invalid_argument("--fp8-fp32-layers requires --fp8-linear true");
    }
    const auto continuous_arguments = result.continuous_slots > 0 ||
                                      !result.continuous_prompt_lengths.empty() ||
                                      !result.continuous_new_token_lengths.empty();
    if ((result.workload == "continuous") != continuous_arguments ||
        (continuous_arguments &&
         (result.continuous_slots <= 0 ||
          result.continuous_prompt_lengths.empty() ||
          result.continuous_new_token_lengths.empty() || !result.use_cache ||
          result.new_tokens != 0))) {
        throw std::invalid_argument(
            "continuous workload requires positive slots, prompt/new-token length lists, cache, and --new-tokens 0");
    }
    if (result.continuous_diagnostics && result.workload != "continuous") {
        throw std::invalid_argument(
            "--continuous-diagnostics requires continuous workload");
    }
    if (!result.continuous_cache_buckets.empty() &&
        (result.workload != "continuous" || result.continuous_diagnostics)) {
        throw std::invalid_argument(
            "--continuous-cache-buckets requires continuous workload without diagnostics");
    }
    if (result.continuous_bucket_overflow &&
        result.continuous_cache_buckets.empty()) {
        throw std::invalid_argument(
            "--continuous-bucket-overflow requires cache buckets");
    }
    if (!result.continuous_prompt_offsets.empty() &&
        result.workload != "continuous") {
        throw std::invalid_argument(
            "--continuous-prompt-offsets requires continuous workload");
    }
    if (!result.continuous_arrival_steps.empty() &&
        result.workload != "continuous") {
        throw std::invalid_argument(
            "--continuous-arrival-steps requires continuous workload");
    }
    if (result.trace_max_elements <= 0) {
        throw std::invalid_argument("--trace-max-elements must be positive");
    }
    if (!result.trace_output.empty() &&
        (result.workload != "prefill" || result.prefill_warmup != 0 ||
         result.prefill_steps != 1)) {
        throw std::invalid_argument(
            "--trace-output requires prefill workload, zero prefill warmup, and one prefill step");
    }
    if (result.trace_all_layer_details && result.trace_output.empty()) {
        throw std::invalid_argument(
            "--trace-all-layer-details requires --trace-output");
    }
    if (!result.trace_value_filter.empty() && result.trace_output.empty()) {
        throw std::invalid_argument(
            "--trace-value-filter requires --trace-output");
    }
    if (result.bf16_algorithm_index < -1 ||
        (result.bf16_algorithm_index >= 0 &&
         (result.workload != "prefill" || !result.bf16_ffn))) {
        throw std::invalid_argument(
            "--bf16-algorithm-index requires BF16 FFN prefill workload");
    }
    if (result.bf16_grouped_qkv_algorithm_index < -1 ||
        (result.bf16_grouped_qkv_algorithm_index >= 0 &&
         (result.device != "hip" || result.workload != "prefill" ||
          !result.bf16_qkv_arena))) {
        throw std::invalid_argument(
            "--bf16-grouped-qkv-algorithm-index requires HIP prefill and QKV Arena");
    }
    if (result.bf16_grouped_qkv_prewarm &&
        result.bf16_grouped_qkv_algorithm_index < 0) {
        throw std::invalid_argument(
            "--bf16-grouped-qkv-prewarm requires an exact grouped algorithm");
    }
    if (result.bf16_grouped_gate_up_algorithm_index < -1 ||
        (result.bf16_grouped_gate_up_algorithm_index >= 0 &&
         (result.device != "hip" || result.workload != "prefill" ||
          !result.bf16_ffn || !result.bf16_ffn_arena))) {
        throw std::invalid_argument(
            "--bf16-grouped-gate-up-algorithm-index requires HIP BF16 FFN Arena prefill");
    }
    if (result.bf16_grouped_gate_up_swish &&
        result.bf16_grouped_gate_up_algorithm_index < 0) {
        throw std::invalid_argument(
            "--bf16-grouped-gate-up-swish requires an exact grouped gate/up algorithm");
    }
    const auto fp32_attention_solution_requested =
        result.fp32_attention_qk_solution_index >= 0 ||
        result.fp32_attention_pv_solution_index >= 0;
    if (result.fp32_attention_qk_solution_index < -1 ||
        result.fp32_attention_pv_solution_index < -1 ||
        (fp32_attention_solution_requested &&
         (result.device != "hip" || result.workload != "prefill"))) {
        throw std::invalid_argument(
            "FP32 Attention solution indices require HIP prefill workload");
    }
    if (!result.cache_logits_output.empty() &&
        (!result.use_cache || result.workload == "prefill" || result.new_tokens < 2)) {
        throw std::invalid_argument(
            "--cache-logits-output requires cached decode with at least two new tokens");
    }
    if (result.allocation_source_diagnostics &&
        (result.workload != "prefill" || result.prefill_warmup != 0 ||
         result.prefill_steps != 1)) {
        throw std::invalid_argument(
            "--allocation-source-diagnostics requires one prefill with zero warmup");
    }
    if (result.strided_copy_diagnostics &&
        (result.workload != "prefill" || result.prefill_warmup != 0 ||
         result.prefill_steps != 1)) {
        throw std::invalid_argument(
            "--strided-copy-diagnostics requires one prefill with zero warmup");
    }
    if (result.inference_bthd_attention &&
        (result.device != "hip" || result.workload != "prefill" ||
         !result.bf16_attention)) {
        throw std::invalid_argument(
            "--inference-bthd-attention requires HIP BF16 Attention prefill");
    }
    if (result.inference_bthd_bf16_qk &&
        (!result.inference_bthd_attention || !result.bf16_qkv_arena ||
         result.bf16_grouped_qkv_algorithm_index < 0)) {
        throw std::invalid_argument(
            "--inference-bthd-bf16-qk requires BTHD Attention, QKV Arena, and an exact grouped QKV algorithm");
    }
    if (result.inference_bthd_online_attention &&
        (!result.inference_bthd_attention || !result.inference_bthd_bf16_qk)) {
        throw std::invalid_argument(
            "--inference-bthd-online-attention requires BTHD Attention and retained BF16 Q/K");
    }
    return result;
}

std::string fp8_compute_policy(const Options& command) {
    const auto weight_name = command.fp8_weight_scale_mode == "device-tensor-amax"
                                 ? "device_tensor_amax_weight"
                                 : command.fp8_weight_scale_mode ==
                                           "output-channel-amax"
                                       ? "output_channel_amax_weight"
                                 : command.fp8_weight_scale_mode == "tensor-amax"
                                       ? "tensor_amax_weight" : "fixed_weight";
    const auto scoped_weight_name =
        command.fp8_weight_scale_mode == "output-channel-amax" &&
                      command.fp8_weight_scale_scope == "attention-output-only"
            ? "attention_output_only_output_channel_amax_weight"
            : weight_name;
    const auto activation_name =
        command.fp8_activation_scale_mode == "ffn-outer-row"
            ? "ffn_outer_row_activation"
            : command.fp8_activation_scale_mode == "tensor-amax"
                  ? "tensor_amax_activation" : "fixed_activation";
    if (command.fp8_diagnostic_mode == "weight-only") {
        return std::string("fp8_e4m3_fnuz_weight_only_diagnostic_") +
               scoped_weight_name;
    }
    if (command.fp8_diagnostic_mode == "activation-only") {
        return std::string("fp8_e4m3_fnuz_activation_only_diagnostic_") +
               activation_name;
    }
    if (command.fp8_diagnostic_mode == "both-roundtrip") {
        return std::string("fp8_e4m3_fnuz_both_roundtrip_diagnostic_") +
               scoped_weight_name + "_" + activation_name;
    }
    if (command.fp8_weight_scale_mode == "output-channel-amax") {
        return std::string("fp8_e4m3_fnuz_") + scoped_weight_name + "_" +
               activation_name;
    }
    if (command.fp8_activation_scale_mode == "ffn-outer-row") {
        return std::string("fp8_e4m3_fnuz_") + scoped_weight_name +
               "_ffn_outer_row";
    }
    if (command.fp8_activation_scale_mode == "tensor-amax" &&
        command.fp8_weight_scale_mode == "tensor-amax") {
        return "fp8_e4m3_fnuz_tensor_amax_weight_activation";
    }
    if (command.fp8_activation_scale_mode == "tensor-amax") {
        return "fp8_e4m3_fnuz_tensor_amax_activation";
    }
    if (command.fp8_weight_scale_mode == "tensor-amax") {
        return "fp8_e4m3_fnuz_tensor_amax_weight";
    }
    if (command.fp8_weight_scale_mode == "device-tensor-amax") {
        return "fp8_e4m3_fnuz_device_tensor_amax_weight";
    }
    return "fp8_e4m3_fnuz_static_scale";
}

std::string fp8_storage_policy(const Options& command) {
    if (command.fp8_diagnostic_mode == "activation-only") {
        return "fp32_linear_weights_with_" + fp8_compute_policy(command);
    }
    return "single_representation_fp8_linear_" + fp8_compute_policy(command);
}

std::string fp8_compute_dtype(const Options& command) {
    if (command.fp8_diagnostic_mode == "weight-only") {
        return "fp32_gemm_with_fp8_roundtrip_weight";
    }
    if (command.fp8_diagnostic_mode == "activation-only") {
        return "fp32_gemm_with_fp8_roundtrip_activation";
    }
    if (command.fp8_diagnostic_mode == "both-roundtrip") {
        return "fp32_gemm_with_fp8_roundtrip_both_operands";
    }
    return "fp8_e4m3_fnuz_linear_with_fp32_boundaries";
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
    const auto batch = state.logits.shape()[0];
    microllm::Tensor history(
        {new_tokens, batch}, microllm::DType::Int32, state.logits.device());
    microllm::Tensor next_tensor;
    if (steady) next_tensor = microllm::ops::argmax_last_dim(state.logits);
    for (std::int64_t generated = 0; generated < new_tokens; ++generated) {
        auto history_slot = history.slice(0, generated, generated + 1)
                                .reshape({batch, 1});
        if (steady) state.logits = model.forward_cached(next_tensor, state.cache);
        microllm::ops::argmax_last_dim_out_(state.logits, history_slot);
        next_tensor = history_slot;
        if (!steady && generated + 1 < new_tokens) {
            state.logits = model.forward_cached(next_tensor, state.cache);
        }
    }
    const auto selected = history.to_int32_vector();
    for (std::int64_t generated = 0; generated < new_tokens; ++generated) {
        const auto offset = static_cast<std::size_t>(generated * batch);
        const auto next = selected[offset];
        if (next < 0 || std::any_of(
                            selected.begin() + static_cast<std::ptrdiff_t>(offset),
                            selected.begin() + static_cast<std::ptrdiff_t>(offset + batch),
                            [next](std::int32_t value) { return value != next; })) {
            throw std::runtime_error(
                "cached identical batch rows produced invalid or different tokens");
        }
        run.suffix.push_back(next);
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

std::vector<std::int64_t> positive_lengths(std::string_view text,
                                           const char* error) {
    const auto parsed = nonnegative_values(text, error);
    std::vector<std::int64_t> result;
    result.reserve(parsed.size());
    for (const auto value : parsed) {
        if (value <= 0) throw std::invalid_argument(error);
        result.push_back(value);
    }
    return result;
}

std::vector<microllm::inference::LengthBucketConfig> cache_buckets(
    std::string_view text) {
    std::vector<microllm::inference::LengthBucketConfig> result;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto item = text.substr(0, comma);
        const auto colon = item.find(':');
        std::int64_t capacity = 0;
        std::int64_t slots = 0;
        const auto capacity_text = item.substr(0, colon);
        const auto slots_text = colon == std::string_view::npos
                                    ? std::string_view{}
                                    : item.substr(colon + 1);
        const auto parsed_capacity = std::from_chars(
            capacity_text.data(), capacity_text.data() + capacity_text.size(),
            capacity);
        const auto parsed_slots = std::from_chars(
            slots_text.data(), slots_text.data() + slots_text.size(), slots);
        if (colon == std::string_view::npos || capacity_text.empty() ||
            slots_text.empty() || parsed_capacity.ec != std::errc{} ||
            parsed_capacity.ptr != capacity_text.data() + capacity_text.size() ||
            parsed_slots.ec != std::errc{} ||
            parsed_slots.ptr != slots_text.data() + slots_text.size() ||
            capacity <= 0 || slots <= 0) {
            throw std::invalid_argument(
                "--continuous-cache-buckets must be capacity:slots pairs");
        }
        result.push_back({.max_sequence_length = capacity,
                          .max_slots = slots});
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
    }
    return result;
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

struct ContinuousOfficialRun {
    std::vector<std::vector<std::int32_t>> generated;
    std::vector<double> request_ttft_ms;
    std::vector<double> request_completion_ms;
    microllm::inference::ContinuousBatchMetrics metrics;
    std::vector<microllm::inference::SelectionDiagnostic> diagnostics;
    bool bucketed_cache = false;
    std::vector<microllm::inference::LengthBucketConfig> cache_buckets;
    std::vector<std::size_t> request_bucket_indices;
    std::int64_t overflow_routed_requests = 0;
};

double percentile(std::vector<double> values, double quantile) {
    if (values.empty() || quantile < 0.0 || quantile > 1.0) {
        throw std::invalid_argument("percentile input is invalid");
    }
    std::sort(values.begin(), values.end());
    const auto position = quantile * static_cast<double>(values.size() - 1U);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const auto fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

ContinuousOfficialRun run_continuous_official(
    microllm::model::TransformerModel& model,
    const std::vector<std::vector<std::int32_t>>& prompts,
    const std::vector<std::int64_t>& new_token_lengths,
    const std::vector<std::int32_t>& arrival_steps,
    std::int64_t slots, const std::vector<microllm::DType>& cache_dtypes,
    bool batch_equal_length_prefill, bool capture_diagnostics,
    const std::vector<microllm::inference::LengthBucketConfig>& buckets,
    bool overflow_to_larger_bucket) {
    if (prompts.size() != new_token_lengths.size() ||
        prompts.size() != arrival_steps.size()) {
        throw std::invalid_argument(
            "continuous prompt and generation counts must match");
    }
    std::int64_t cache_capacity = 0;
    for (std::size_t index = 0; index < prompts.size(); ++index) {
        cache_capacity = std::max(
            cache_capacity,
            static_cast<std::int64_t>(prompts[index].size()) +
                new_token_lengths[index]);
    }
    ContinuousOfficialRun result;
    const auto execute = [&](auto& scheduler) {
        std::vector<microllm::inference::RequestId> ids(prompts.size(), 0);
        std::size_t submitted = 0;
        std::int64_t arrival_clock = 0;
        while (submitted < prompts.size() || scheduler.has_active_requests()) {
            for (std::size_t index = 0; index < prompts.size(); ++index) {
                if (ids[index] != 0 || arrival_steps[index] > arrival_clock) {
                    continue;
                }
                ids[index] = scheduler.submit(
                    prompts[index],
                    {.max_new_tokens = new_token_lengths[index],
                     .temperature = 0.0F,
                     .top_k = 1,
                     .seed = static_cast<std::uint64_t>(index + 1),
                     .kv_cache_dtype = cache_dtypes.front(),
                     .kv_cache_layer_dtypes = cache_dtypes,
                     .stop_tokens = {}});
                ++submitted;
            }
            if (scheduler.has_active_requests()) scheduler.step();
            ++arrival_clock;
        }
        result.generated.reserve(ids.size());
        result.request_ttft_ms.reserve(ids.size());
        result.request_completion_ms.reserve(ids.size());
        for (const auto id : ids) {
            const auto snapshot = scheduler.request(id);
            if (snapshot.time_to_first_token_ms < 0.0 ||
                snapshot.completion_latency_ms <
                    snapshot.time_to_first_token_ms) {
                throw std::runtime_error(
                    "continuous request latency lifecycle is incomplete");
            }
            result.generated.push_back(snapshot.generated);
            result.request_ttft_ms.push_back(snapshot.time_to_first_token_ms);
            result.request_completion_ms.push_back(
                snapshot.completion_latency_ms);
        }
        return ids;
    };
    if (buckets.empty()) {
        microllm::inference::ContinuousBatchScheduler scheduler(
            model, {.max_slots = slots,
                    .max_sequence_length = cache_capacity,
                    .kv_cache_dtype = cache_dtypes.front(),
                    .kv_cache_layer_dtypes = cache_dtypes,
                    .batch_equal_length_prefill = batch_equal_length_prefill,
                    .capture_selection_diagnostics = capture_diagnostics});
        (void)execute(scheduler);
        result.metrics = scheduler.metrics();
        result.diagnostics = scheduler.selection_diagnostics();
    } else {
        microllm::inference::LengthBucketedBatchScheduler scheduler(
            model, {.buckets = buckets,
                    .kv_cache_dtype = cache_dtypes.front(),
                    .kv_cache_layer_dtypes = cache_dtypes,
                    .batch_equal_length_prefill = batch_equal_length_prefill,
                    .overflow_to_larger_bucket = overflow_to_larger_bucket});
        const auto ids = execute(scheduler);
        const auto bucketed = scheduler.metrics();
        result.bucketed_cache = true;
        result.cache_buckets = buckets;
        result.metrics.scheduler_steps = bucketed.scheduler_steps;
        result.metrics.occupied_slot_steps = bucketed.occupied_slot_steps;
        result.metrics.occupied_slots = bucketed.occupied_slots;
        result.metrics.peak_occupied_slots = bucketed.peak_occupied_slots;
        result.metrics.slot_utilization = bucketed.slot_utilization;
        result.metrics.allocated_cache_bytes = bucketed.allocated_cache_bytes;
        result.metrics.active_cache_bytes = bucketed.active_cache_bytes;
        result.metrics.peak_active_cache_bytes =
            bucketed.peak_active_cache_bytes;
        result.overflow_routed_requests = bucketed.overflow_routed_requests;
        for (const auto& child : bucketed.buckets) {
            result.metrics.submitted_requests += child.submitted_requests;
            result.metrics.completed_requests += child.completed_requests;
            result.metrics.cancelled_requests += child.cancelled_requests;
            result.metrics.stop_completed_requests +=
                child.stop_completed_requests;
            result.metrics.slot_admissions += child.slot_admissions;
            result.metrics.slot_refills += child.slot_refills;
            result.metrics.row_prefill_calls += child.row_prefill_calls;
            result.metrics.prefill_batch_calls += child.prefill_batch_calls;
            result.metrics.batched_prefill_calls += child.batched_prefill_calls;
            result.metrics.batched_prefill_rows += child.batched_prefill_rows;
            result.metrics.batch_decode_calls += child.batch_decode_calls;
            result.metrics.uniform_batch_decode_calls +=
                child.uniform_batch_decode_calls;
            result.metrics.divergent_batch_decode_calls +=
                child.divergent_batch_decode_calls;
            result.metrics.compacted_batch_decode_calls +=
                child.compacted_batch_decode_calls;
            result.metrics.positions_aware_batch_decode_calls +=
                child.positions_aware_batch_decode_calls;
            result.metrics.logical_decode_rows += child.logical_decode_rows;
            result.metrics.dummy_decode_rows += child.dummy_decode_rows;
            result.metrics.inactive_rows_skipped +=
                child.inactive_rows_skipped;
            result.metrics.selection_calls += child.selection_calls;
        }
        result.request_bucket_indices.reserve(ids.size());
        for (const auto id : ids) {
            result.request_bucket_indices.push_back(
                scheduler.request_bucket(id));
        }
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto command = options(argc, argv);
        auto external = microllm::model::load_huggingface_config(command.config);
        if (command.fp8_linear) {
            external.model.linear_precision =
                microllm::model::LinearPrecision::Float8E4M3FNUZ;
            external.model.fp8_activation_scale = command.fp8_activation_scale;
            external.model.fp8_activation_minimum_scale =
                command.fp8_activation_minimum_scale;
            external.model.fp8_weight_scale = command.fp8_weight_scale;
            external.model.fp8_weight_scale_mode =
                command.fp8_weight_scale_mode == "tensor-amax"
                    ? microllm::model::Fp8WeightScaleMode::TensorAmax
                    : command.fp8_weight_scale_mode == "device-tensor-amax"
                    ? microllm::model::Fp8WeightScaleMode::DeviceTensorAmax
                    : command.fp8_weight_scale_mode == "output-channel-amax"
                    ? microllm::model::Fp8WeightScaleMode::OutputChannelAmax
                    : microllm::model::Fp8WeightScaleMode::Fixed;
            external.model.fp8_weight_scale_scope =
                command.fp8_weight_scale_scope == "attention-output-only"
                    ? microllm::model::Fp8WeightScaleScope::AttentionOutputOnly
                    : microllm::model::Fp8WeightScaleScope::AllLinear;
            external.model.fp8_activation_scale_mode =
                command.fp8_activation_scale_mode == "tensor-amax"
                    ? microllm::model::Fp8ActivationScaleMode::TensorAmax
                    : command.fp8_activation_scale_mode == "ffn-outer-row"
                    ? microllm::model::Fp8ActivationScaleMode::FfnOuterRow
                    : microllm::model::Fp8ActivationScaleMode::Fixed;
            external.model.fp8_diagnostic_mode =
                command.fp8_diagnostic_mode == "weight-only"
                    ? microllm::model::Fp8DiagnosticMode::WeightOnly
                    : command.fp8_diagnostic_mode == "activation-only"
                    ? microllm::model::Fp8DiagnosticMode::ActivationOnly
                    : command.fp8_diagnostic_mode == "both-roundtrip"
                    ? microllm::model::Fp8DiagnosticMode::BothRoundtrip
                    : microllm::model::Fp8DiagnosticMode::Full;
            for (const auto layer : nonnegative_values(
                     command.fp8_fp32_layers,
                     "--fp8-fp32-layers must be comma-separated nonnegative indices")) {
                external.model.fp8_fp32_layers.push_back(layer);
            }
        }
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
        microllm::model::Fp8WeightPreparationReport fp8_report;
        microllm::model::Bf16GroupedQkvPrewarmReport grouped_qkv_prewarm_report;
        microllm::runtime::reset_allocation_peak(device);
        const auto preparation_start = std::chrono::steady_clock::now();
        if (command.bf16_ffn) bf16_report = model.prepare_bf16_ffn_inference();
        if (command.bf16_ffn_arena) {
            model.set_bf16_ffn_arena_enabled(
                true, command.bf16_ffn_arena_minimum_rows);
        }
        model.set_bf16_ffn_norm_fusion_enabled(
            command.bf16_ffn_norm_fusion);
        if (command.bf16_attention) {
            bf16_attention_report = model.prepare_bf16_attention_inference();
        }
        microllm::ops::enable_inference_bthd_attention(
            command.inference_bthd_attention);
        microllm::ops::enable_inference_bthd_bf16_qk(
            command.inference_bthd_bf16_qk);
        microllm::ops::enable_inference_bthd_online_attention(
            command.inference_bthd_online_attention);
        if (command.bf16_qkv_arena) {
            model.set_bf16_qkv_arena_enabled(
                true, command.bf16_qkv_arena_minimum_rows);
        }
        model.set_bf16_attention_norm_fusion_enabled(
            command.bf16_attention_norm_fusion);
        if (command.attention_core_arena) {
            model.set_attention_core_arena_enabled(
                true, command.attention_core_arena_minimum_sequence);
        }
        model.set_cached_attention_split_sequence(
            command.cached_attention_splits,
            command.cached_attention_minimum_sequence);
        if (command.fp8_linear) {
            fp8_report = model.prepare_fp8_inference_weights();
            microllm::ops::clear_fp8_dispatch_registry();
            microllm::ops::clear_fp8_dynamic_quant_stats();
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
        if (command.bf16_algorithm_index >= 0) {
            microllm::ops::clear_bf16_algorithm_registry();
            microllm::ops::register_bf16_algorithm(
                command.batch * static_cast<std::int64_t>(ids.size()),
                external.model.dimension, external.model.ffn_dimension,
                microllm::DType::BFloat16,
                command.bf16_algorithm_index);
        }
        if (command.bf16_grouped_gate_up_algorithm_index >= 0) {
            microllm::ops::clear_bf16_grouped_gate_up_registry();
            const auto key =
                microllm::ops::make_bf16_grouped_gate_up_key(
                    command.batch *
                        static_cast<std::int64_t>(ids.size()),
                    external.model.dimension,
                    external.model.ffn_dimension, device);
            microllm::ops::register_bf16_grouped_gate_up_algorithm(
                key, command.bf16_grouped_gate_up_algorithm_index);
        }
        microllm::ops::enable_bf16_grouped_gate_up_swish(
            command.bf16_grouped_gate_up_swish);
        if (command.bf16_grouped_qkv_algorithm_index >= 0) {
            microllm::ops::clear_bf16_grouped_qkv_registry();
            const auto key = microllm::ops::make_bf16_grouped_qkv_key(
                command.batch * static_cast<std::int64_t>(ids.size()),
                external.model.dimension, external.model.dimension,
                external.model.kv_dimension(), external.model.kv_dimension(),
                device);
            microllm::ops::register_bf16_grouped_qkv_algorithm(
                key, command.bf16_grouped_qkv_algorithm_index);
            if (command.bf16_grouped_qkv_prewarm) {
                grouped_qkv_prewarm_report =
                    model.prewarm_bf16_grouped_qkv(
                        command.batch * static_cast<std::int64_t>(ids.size()));
            }
        }
        if (command.fp32_attention_qk_solution_index >= 0 ||
            command.fp32_attention_pv_solution_index >= 0) {
            microllm::ops::clear_fp32_matmul_solution_registry();
            const auto sequence = static_cast<std::int64_t>(ids.size());
            const auto heads = external.model.heads;
            const auto width = external.model.head_dimension();
            const microllm::Shape qkv_shape{
                command.batch, heads, sequence, width};
            if (command.fp32_attention_qk_solution_index >= 0) {
                const auto key = microllm::ops::make_fp32_matmul_solution_key(
                    qkv_shape, qkv_shape, device, false, true);
                microllm::ops::register_fp32_matmul_solution(
                    key, command.fp32_attention_qk_solution_index);
            }
            if (command.fp32_attention_pv_solution_index >= 0) {
                const microllm::Shape probability_shape{
                    command.batch, heads, sequence, sequence};
                const auto key = microllm::ops::make_fp32_matmul_solution_key(
                    probability_shape, qkv_shape, device, false, false);
                microllm::ops::register_fp32_matmul_solution(
                    key, command.fp32_attention_pv_solution_index);
            }
        }
        if (command.workload == "continuous") {
            const auto prompt_lengths = positive_lengths(
                command.continuous_prompt_lengths,
                "--continuous-prompt-lengths must contain positive comma-separated lengths");
            const auto new_token_lengths = positive_lengths(
                command.continuous_new_token_lengths,
                "--continuous-new-token-lengths must contain positive comma-separated lengths");
            if (prompt_lengths.empty() ||
                prompt_lengths.size() != new_token_lengths.size()) {
                throw std::invalid_argument(
                    "continuous prompt and new-token lists must have equal nonzero length");
            }
            std::vector<std::int32_t> arrival_steps;
            if (command.continuous_arrival_steps.empty()) {
                arrival_steps.assign(prompt_lengths.size(), 0);
            } else {
                arrival_steps = nonnegative_values(
                    command.continuous_arrival_steps,
                    "--continuous-arrival-steps must contain nonnegative comma-separated steps");
                if (arrival_steps.size() != prompt_lengths.size()) {
                    throw std::invalid_argument(
                        "continuous arrival steps must match request count");
                }
            }
            const auto continuous_buckets =
                cache_buckets(command.continuous_cache_buckets);
            if (!continuous_buckets.empty()) {
                const auto bucket_slots = std::accumulate(
                    continuous_buckets.begin(), continuous_buckets.end(),
                    std::int64_t{0}, [](std::int64_t total, const auto& bucket) {
                        return total + bucket.max_slots;
                    });
                if (bucket_slots != command.continuous_slots) {
                    throw std::invalid_argument(
                        "continuous cache bucket slots must sum to --continuous-slots");
                }
            }
            std::vector<std::int32_t> prompt_offsets;
            if (command.continuous_prompt_offsets.empty()) {
                prompt_offsets.resize(prompt_lengths.size());
                std::iota(prompt_offsets.begin(), prompt_offsets.end(), 0);
            } else {
                prompt_offsets = nonnegative_values(
                    command.continuous_prompt_offsets,
                    "--continuous-prompt-offsets must contain nonnegative comma-separated offsets");
                if (prompt_offsets.size() != prompt_lengths.size()) {
                    throw std::invalid_argument(
                        "continuous prompt offsets must match request count");
                }
            }
            std::vector<std::vector<std::int32_t>> prompts;
            prompts.reserve(prompt_lengths.size());
            for (std::size_t request = 0; request < prompt_lengths.size(); ++request) {
                if (prompt_lengths[request] + new_token_lengths[request] >
                    external.model.max_sequence_length) {
                    throw std::invalid_argument(
                        "continuous request exceeds model context");
                }
                std::vector<std::int32_t> prompt(
                    static_cast<std::size_t>(prompt_lengths[request]));
                for (std::size_t index = 0; index < prompt.size(); ++index) {
                    prompt[index] = ids[
                        (index + static_cast<std::size_t>(
                                     prompt_offsets[request])) % ids.size()];
                }
                prompts.push_back(std::move(prompt));
            }
            const auto warmup_start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < command.warmup; ++iteration) {
                (void)run_continuous_official(
                    model, prompts, new_token_lengths, arrival_steps,
                    command.continuous_slots, cache_dtypes,
                    command.continuous_prefill_batch, false,
                    continuous_buckets,
                    command.continuous_bucket_overflow);
            }
            microllm::runtime::synchronize(device);
            const auto warmup_finish = std::chrono::steady_clock::now();
            if (device.is_hip()) {
                microllm::runtime::enable_hip_caching_allocator(device);
            }
            microllm::runtime::reset_allocation_peak(device);
            microllm::runtime::reset_transfer_stats();
            double measured_ms = 0.0;
            ContinuousOfficialRun last;
            std::vector<std::vector<std::int32_t>> expected;
            for (int iteration = 0; iteration < command.steps; ++iteration) {
                const auto start = std::chrono::steady_clock::now();
                auto current = run_continuous_official(
                    model, prompts, new_token_lengths, arrival_steps,
                    command.continuous_slots, cache_dtypes,
                    command.continuous_prefill_batch,
                    command.continuous_diagnostics, continuous_buckets,
                    command.continuous_bucket_overflow);
                microllm::runtime::synchronize(device);
                const auto finish = std::chrono::steady_clock::now();
                measured_ms += std::chrono::duration<double, std::milli>(
                                   finish - start).count();
                if (iteration == 0) {
                    expected = current.generated;
                } else if (current.generated != expected) {
                    throw std::runtime_error(
                        "continuous official generation changed across measured runs");
                }
                last = std::move(current);
            }
            std::int64_t tokens_per_run = 0;
            std::uint64_t checksum = 0;
            for (std::size_t request = 0; request < last.generated.size(); ++request) {
                if (last.generated[request].size() !=
                    static_cast<std::size_t>(new_token_lengths[request])) {
                    throw std::runtime_error(
                        "continuous official generation returned the wrong length");
                }
                tokens_per_run += new_token_lengths[request];
                for (const auto token : last.generated[request]) {
                    checksum = checksum * 131U +
                               static_cast<std::uint64_t>(token);
                }
            }
            const auto measured_tokens = tokens_per_run * command.steps;
            const auto allocation = microllm::runtime::allocation_stats(device);
            const auto transfers = microllm::runtime::transfer_stats();
            const auto info = device.is_cpu()
                                  ? microllm::runtime::DeviceInfo{
                                        device, "host CPU", "host"}
                                  : microllm::runtime::device_info(device);
            const auto resident_weight_bytes =
                external.model.weight_bytes(sizeof(float)) -
                bf16_report.fp32_bytes_released +
                bf16_report.bf16_bytes_retained -
                bf16_attention_report.fp32_bytes_released +
                bf16_attention_report.bf16_bytes_retained -
                fp8_report.fp32_bytes_released +
                fp8_report.fp8_bytes_retained +
                fp8_report.scale_bytes_retained;
            std::cout << std::setprecision(9)
                      << "{\"schema_version\":1,\"status\":\"pass\""
                      << ",\"record_type\":\"official_continuous_serving_measurement\""
                      << ",\"device\":\"" << device.str() << "\""
                      << ",\"device_name\":\"" << info.name << "\""
                      << ",\"architecture\":\"" << info.architecture << "\""
                      << ",\"device_total_bytes\":" << info.total_memory
                      << ",\"parameter_count\":" << model.parameter_count()
                      << ",\"loaded_tensors\":" << report.loaded.size()
                      << ",\"resident_weight_bytes\":" << resident_weight_bytes
                      << ",\"linear_precision_policy\":\""
                      << (command.fp8_linear
                              ? fp8_compute_policy(command)
                                             : command.bf16_attention
                                                   ? "bf16_ffn_attention"
                                                   : command.bf16_ffn ? "bf16_ffn"
                                                                      : "fp32")
                      << "\""
                      << ",\"fp8_activation_scale\":"
                      << command.fp8_activation_scale
                      << ",\"fp8_activation_minimum_scale\":"
                      << command.fp8_activation_minimum_scale
                      << ",\"fp8_weight_scale\":"
                      << command.fp8_weight_scale
                      << ",\"fp8_weight_scale_mode\":\""
                      << command.fp8_weight_scale_mode << "\""
                      << ",\"fp8_weight_scale_scope\":\""
                      << command.fp8_weight_scale_scope << "\""
                      << ",\"fp8_activation_scale_mode\":\""
                      << command.fp8_activation_scale_mode << "\""
                      << ",\"fp8_diagnostic_mode\":\""
                      << command.fp8_diagnostic_mode << "\""
                      << ",\"fp8_fp32_layers\":\""
                      << command.fp8_fp32_layers << "\""
                      << ",\"fp8_weight_scale_min\":"
                      << fp8_report.minimum_weight_scale
                      << ",\"fp8_weight_scale_max\":"
                      << fp8_report.maximum_weight_scale
                      << ",\"fp8_weight_bytes_scanned\":"
                      << fp8_report.weight_bytes_scanned
                      << ",\"fp8_device_weight_bytes_scanned\":"
                      << fp8_report.device_weight_bytes_scanned
                      << ",\"fp8_device_amax_tensors\":"
                      << fp8_report.device_amax_tensors
                      << ",\"fp8_host_scale_summary_available\":"
                      << (fp8_report.host_scale_summary_available ? "true" : "false")
                      << ",\"fp8_converted_tensors\":"
                      << fp8_report.converted_tensors
                      << ",\"fp8_linears_covered\":"
                      << fp8_report.linears_covered
                      << ",\"fp8_native_shapes\":"
                      << microllm::ops::fp8_dispatch_stats().native_shapes
                      << ",\"fp8_software_fallback_shapes\":"
                      << microllm::ops::fp8_dispatch_stats().software_fallback_shapes
                      << ",\"fp8_software_fallback_calls\":"
                      << microllm::ops::fp8_dispatch_stats().software_fallback_calls
                      << ",\"fp8_outer_row_fallback_calls\":"
                      << microllm::ops::fp8_dispatch_stats().outer_row_fallback_calls
                      << ",\"fp8_outer_row_native_status\":"
                      << microllm::ops::fp8_dispatch_stats().outer_row_native_status
                      << ",\"fp8_output_column_scale_calls\":"
                      << microllm::ops::fp8_dispatch_stats().output_column_scale_calls
                      << ",\"fp8_output_column_native_status\":"
                      << microllm::ops::fp8_dispatch_stats().output_column_native_status
                      << ",\"fp8_dynamic_tensor_calls\":"
                      << microllm::ops::fp8_dynamic_quant_stats().tensor_calls
                      << ",\"fp8_dynamic_row_calls\":"
                      << microllm::ops::fp8_dynamic_quant_stats().row_calls
                      << ",\"fp8_dynamic_tensor_elements\":"
                      << microllm::ops::fp8_dynamic_quant_stats().tensor_elements
                      << ",\"fp8_dynamic_row_elements\":"
                      << microllm::ops::fp8_dynamic_quant_stats().row_elements
                      << ",\"fp8_dynamic_column_calls\":"
                      << microllm::ops::fp8_dynamic_quant_stats().column_calls
                      << ",\"fp8_dynamic_column_elements\":"
                      << microllm::ops::fp8_dynamic_quant_stats().column_elements
                      << ",\"fp8_dynamic_clipped_tensor_calls\":"
                      << microllm::ops::fp8_dynamic_quant_stats().clipped_tensor_calls
                      << ",\"request_count\":" << prompts.size()
                      << ",\"continuous_slots\":" << command.continuous_slots
                      << ",\"bucketed_cache\":"
                      << (last.bucketed_cache ? "true" : "false")
                      << ",\"continuous_bucket_overflow\":"
                      << (command.continuous_bucket_overflow ? "true" : "false")
                      << ",\"overflow_routed_requests\":"
                      << last.overflow_routed_requests
                      << ",\"continuous_prefill_batch\":"
                      << (command.continuous_prefill_batch ? "true" : "false")
                      << ",\"continuous_diagnostics\":"
                      << (command.continuous_diagnostics ? "true" : "false")
                      << ",\"warmup\":" << command.warmup
                      << ",\"steps\":" << command.steps
                      << ",\"warmup_ms\":"
                      << std::chrono::duration<double, std::milli>(
                             warmup_finish - warmup_start).count()
                      << ",\"measured_ms\":" << measured_ms
                      << ",\"measured_tokens\":" << measured_tokens
                      << ",\"tokens_per_second\":"
                      << static_cast<double>(measured_tokens) * 1000.0 /
                             measured_ms
                      << ",\"scheduler_steps\":"
                      << last.metrics.scheduler_steps
                      << ",\"slot_admissions\":"
                      << last.metrics.slot_admissions
                      << ",\"slot_refills\":" << last.metrics.slot_refills
                      << ",\"row_prefill_calls\":"
                      << last.metrics.row_prefill_calls
                      << ",\"prefill_batch_calls\":"
                      << last.metrics.prefill_batch_calls
                      << ",\"batched_prefill_calls\":"
                      << last.metrics.batched_prefill_calls
                      << ",\"batched_prefill_rows\":"
                      << last.metrics.batched_prefill_rows
                      << ",\"batch_decode_calls\":"
                      << last.metrics.batch_decode_calls
                      << ",\"positions_aware_batch_decode_calls\":"
                      << last.metrics.positions_aware_batch_decode_calls
                      << ",\"uniform_batch_decode_calls\":"
                      << last.metrics.uniform_batch_decode_calls
                      << ",\"inactive_rows_skipped\":"
                      << last.metrics.inactive_rows_skipped
                      << ",\"slot_utilization\":"
                      << last.metrics.slot_utilization
                      << ",\"allocated_cache_bytes\":"
                      << last.metrics.allocated_cache_bytes
                      << ",\"peak_active_cache_bytes\":"
                      << last.metrics.peak_active_cache_bytes
                      << ",\"kv_cache_byte_utilization\":"
                      << (last.metrics.allocated_cache_bytes == 0
                              ? 0.0
                              : static_cast<double>(
                                    last.metrics.peak_active_cache_bytes) /
                                    static_cast<double>(
                                        last.metrics.allocated_cache_bytes))
                      << ",\"request_ttft_p50_ms\":"
                      << percentile(last.request_ttft_ms, 0.50)
                      << ",\"request_ttft_p95_ms\":"
                      << percentile(last.request_ttft_ms, 0.95)
                      << ",\"request_completion_p50_ms\":"
                      << percentile(last.request_completion_ms, 0.50)
                      << ",\"request_completion_p95_ms\":"
                      << percentile(last.request_completion_ms, 0.95)
                      << ",\"engine_peak_bytes\":" << allocation.peak_bytes
                      << ",\"engine_backend_allocation_calls\":"
                      << allocation.backend_allocation_calls
                      << ",\"engine_cache_reuse_calls\":"
                      << allocation.cache_reuse_calls
                      << ",\"measured_h2d_calls\":"
                      << transfers.host_to_device_calls
                      << ",\"measured_h2d_bytes\":"
                      << transfers.host_to_device_bytes
                      << ",\"measured_d2h_calls\":"
                      << transfers.device_to_host_calls
                      << ",\"measured_d2h_bytes\":"
                      << transfers.device_to_host_bytes
                      << ",\"measured_d2d_calls\":"
                      << transfers.device_to_device_calls
                      << ",\"measured_d2d_bytes\":"
                      << transfers.device_to_device_bytes
                      << ",\"prompt_lengths\":[";
            for (std::size_t index = 0; index < prompt_lengths.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << prompt_lengths[index];
            }
            std::cout << "],\"new_token_lengths\":[";
            for (std::size_t index = 0; index < new_token_lengths.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << new_token_lengths[index];
            }
            std::cout << "],\"continuous_cache_buckets\":[";
            for (std::size_t index = 0;
                 index < last.cache_buckets.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << "{\"max_sequence_length\":"
                          << last.cache_buckets[index].max_sequence_length
                          << ",\"max_slots\":"
                          << last.cache_buckets[index].max_slots << '}';
            }
            std::cout << "],\"request_bucket_indices\":[";
            for (std::size_t index = 0;
                 index < last.request_bucket_indices.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << last.request_bucket_indices[index];
            }
            std::cout << "],\"arrival_steps\":[";
            for (std::size_t index = 0; index < arrival_steps.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << arrival_steps[index];
            }
            std::cout << "],\"prompt_offsets\":[";
            for (std::size_t index = 0; index < prompt_offsets.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << prompt_offsets[index];
            }
            std::cout << "],\"request_ttft_ms\":[";
            for (std::size_t index = 0;
                 index < last.request_ttft_ms.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << last.request_ttft_ms[index];
            }
            std::cout << "],\"request_completion_ms\":[";
            for (std::size_t index = 0;
                 index < last.request_completion_ms.size(); ++index) {
                if (index != 0) std::cout << ',';
                std::cout << last.request_completion_ms[index];
            }
            std::cout << "],\"deterministic_across_steps\":true"
                      << ",\"generated_tokens\":[";
            for (std::size_t request = 0; request < last.generated.size(); ++request) {
                if (request != 0) std::cout << ',';
                std::cout << '[';
                for (std::size_t index = 0;
                     index < last.generated[request].size(); ++index) {
                    if (index != 0) std::cout << ',';
                    std::cout << last.generated[request][index];
                }
                std::cout << ']';
            }
            std::cout << "],\"selection_diagnostic_count\":"
                      << last.diagnostics.size()
                      << ",\"selection_diagnostics\":[";
            for (std::size_t index = 0; index < last.diagnostics.size(); ++index) {
                if (index != 0) std::cout << ',';
                const auto& diagnostic = last.diagnostics[index];
                std::cout << "{\"scheduler_step\":"
                          << diagnostic.scheduler_step
                          << ",\"request_id\":" << diagnostic.request_id
                          << ",\"slot\":" << diagnostic.slot
                          << ",\"generated_index\":"
                          << diagnostic.generated_index
                          << ",\"cache_position\":"
                          << diagnostic.cache_position
                          << ",\"logit_batch_size\":"
                          << diagnostic.logit_batch_size
                          << ",\"logit_source\":\""
                          << diagnostic.logit_source << "\""
                          << ",\"device_selected_token\":"
                          << diagnostic.device_selected_token
                          << ",\"top1_token\":" << diagnostic.top1_token
                          << ",\"top1_logit\":" << diagnostic.top1_logit
                          << ",\"top2_token\":" << diagnostic.top2_token
                          << ",\"top2_logit\":" << diagnostic.top2_logit
                          << ",\"top1_top2_margin\":"
                          << diagnostic.top1_top2_margin
                          << ",\"device_argmax_matches_top1\":"
                          << (diagnostic.device_argmax_matches_top1
                                  ? "true" : "false")
                          << '}';
            }
            std::cout << "],\"token_checksum\":" << checksum << "}\n";
            return 0;
        }
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
        std::size_t trace_record_count = 0;
        microllm::runtime::AllocationSourceDiagnostics allocation_sources;
        microllm::runtime::StridedCopyDiagnostics strided_copies;
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
            if (command.allocation_source_diagnostics) {
                microllm::runtime::reset_allocation_source_diagnostics();
                microllm::runtime::enable_allocation_source_diagnostics(true);
            }
            if (command.strided_copy_diagnostics) {
                microllm::runtime::reset_strided_copy_diagnostics();
                microllm::runtime::enable_strided_copy_diagnostics(true);
            }
            std::unique_ptr<microllm::profiling::TraceSession> trace_session;
            std::unique_ptr<microllm::profiling::ScopedTraceSession> trace_scope;
            if (!command.trace_output.empty()) {
                microllm::profiling::TraceOptions trace_options;
                trace_options.phase = "inference-values";
                trace_options.record_operators = false;
                trace_options.record_layers = true;
                trace_options.record_model = true;
                trace_options.capture_values = true;
                trace_options.synchronize_device = true;
                trace_options.record_all_layer_details =
                    command.trace_all_layer_details;
                auto filters = command.trace_value_filter;
                while (!filters.empty()) {
                    const auto comma = filters.find(',');
                    const auto value = filters.substr(0, comma);
                    if (value.empty()) {
                        throw std::invalid_argument(
                            "--trace-value-filter contains an empty item");
                    }
                    trace_options.value_name_filters.push_back(value);
                    if (comma == std::string::npos) break;
                    filters.erase(0, comma + 1);
                }
                trace_options.max_captured_elements =
                    static_cast<std::size_t>(command.trace_max_elements);
                trace_session = std::make_unique<microllm::profiling::TraceSession>(
                    "microllm", "hf-prefill", trace_options);
                trace_scope =
                    std::make_unique<microllm::profiling::ScopedTraceSession>(
                        *trace_session);
            }
            const auto forward_start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < command.prefill_steps; ++iteration) {
                logits_tensor = prefill();
            }
            microllm::runtime::synchronize(device);
            if (command.allocation_source_diagnostics) {
                microllm::runtime::enable_allocation_source_diagnostics(false);
                allocation_sources =
                    microllm::runtime::allocation_source_diagnostics();
            }
            if (command.strided_copy_diagnostics) {
                microllm::runtime::enable_strided_copy_diagnostics(false);
                strided_copies =
                    microllm::runtime::strided_copy_diagnostics();
            }
            const auto forward_finish = std::chrono::steady_clock::now();
            trace_scope.reset();
            if (trace_session != nullptr) {
                trace_session->write_jsonl(command.trace_output);
                trace_record_count = trace_session->records().size();
            }
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
            const auto batch_rows = static_cast<std::size_t>(command.batch);
            const auto expected = command.prefill_logits_mode == "last"
                                      ? batch_rows * vocabulary
                                      : batch_rows * ids.size() * vocabulary;
            if (logits.size() != expected) {
                throw std::runtime_error(
                    "prefill logits shape does not match batch export contract");
            }
            if (command.prefill_logits_mode == "last") {
                output.write(
                    reinterpret_cast<const char*>(logits.data()),
                    static_cast<std::streamsize>(
                        logits.size() * sizeof(float)));
            } else {
                for (std::size_t row = 0; row < batch_rows; ++row) {
                    const auto row_offset =
                        (row * ids.size() + ids.size() - 1U) * vocabulary;
                    output.write(
                        reinterpret_cast<const char*>(
                            logits.data() + row_offset),
                        static_cast<std::streamsize>(
                            vocabulary * sizeof(float)));
                }
            }
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
        const auto bf16_arena_stats = model.bf16_ffn_arena_stats();
        const auto bf16_qkv_arena_stats = model.bf16_qkv_arena_stats();
        const auto attention_core_arena_stats =
            model.attention_core_arena_stats();
        const auto measured_transfers = microllm::runtime::transfer_stats();
        const auto fp32_solution_stats =
            microllm::ops::fp32_matmul_solution_stats();
        const auto grouped_qkv_stats =
            microllm::ops::bf16_grouped_qkv_stats();
        const auto grouped_gate_up_stats =
            microllm::ops::bf16_grouped_gate_up_stats();
        const auto online_attention_native_calls =
            microllm::ops::rocwmma_online_attention_native_calls();
        const auto online_attention_fallback_calls =
            microllm::ops::rocwmma_online_attention_fallback_calls();
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
                  << ",\"trace_record_count\":"
                  << trace_record_count
                  << ",\"inference_bthd_attention\":"
                  << (command.inference_bthd_attention ? "true" : "false")
                  << ",\"inference_bthd_online_attention\":"
                  << (command.inference_bthd_online_attention ? "true" : "false")
                  << ",\"inference_bthd_bf16_qk\":"
                  << (command.inference_bthd_bf16_qk ? "true" : "false")
                  << ",\"rocwmma_online_attention_native_calls\":"
                  << online_attention_native_calls
                  << ",\"rocwmma_online_attention_fallback_calls\":"
                  << online_attention_fallback_calls
                  << ",\"bf16_algorithm_index\":"
                  << command.bf16_algorithm_index
                  << ",\"bf16_grouped_qkv_algorithm_index\":"
                  << command.bf16_grouped_qkv_algorithm_index
                  << ",\"bf16_grouped_gate_up_algorithm_index\":"
                  << command.bf16_grouped_gate_up_algorithm_index
                  << ",\"bf16_grouped_gate_up_swish\":"
                  << (command.bf16_grouped_gate_up_swish ? "true" : "false")
                  << ",\"bf16_grouped_qkv_prewarm\":"
                  << (command.bf16_grouped_qkv_prewarm ? "true" : "false")
                  << ",\"bf16_grouped_qkv_prewarm_rows\":"
                  << grouped_qkv_prewarm_report.rows
                  << ",\"bf16_grouped_qkv_prewarm_blocks\":"
                  << grouped_qkv_prewarm_report.blocks
                  << ",\"bf16_grouped_qkv_prewarm_ms\":"
                  << grouped_qkv_prewarm_report.total_ms
                  << ",\"bf16_grouped_qkv_prewarm_kernel_ms\":"
                  << grouped_qkv_prewarm_report.kernel_setup_ms
                  << ",\"bf16_grouped_qkv_prewarm_arguments_ms\":"
                  << grouped_qkv_prewarm_report.argument_setup_ms
                  << ",\"bf16_grouped_qkv_already_warm\":"
                  << (grouped_qkv_prewarm_report.already_warm ? "true" : "false")
                  << ",\"bf16_grouped_qkv_registered_entries\":"
                  << grouped_qkv_stats.registered_entries
                  << ",\"bf16_grouped_qkv_algorithm_entries\":"
                  << grouped_qkv_stats.algorithm_entries
                  << ",\"bf16_grouped_qkv_algorithm_hits\":"
                  << grouped_qkv_stats.algorithm_hits
                  << ",\"bf16_grouped_qkv_algorithm_misses\":"
                  << grouped_qkv_stats.algorithm_misses
                  << ",\"bf16_grouped_qkv_kernel_entries\":"
                  << grouped_qkv_stats.kernel_entries
                  << ",\"bf16_grouped_qkv_kernel_hits\":"
                  << grouped_qkv_stats.kernel_hits
                  << ",\"bf16_grouped_qkv_kernel_misses\":"
                  << grouped_qkv_stats.kernel_misses
                  << ",\"bf16_grouped_qkv_plan_entries\":"
                  << grouped_qkv_stats.plan_entries
                  << ",\"bf16_grouped_qkv_plan_hits\":"
                  << grouped_qkv_stats.plan_hits
                  << ",\"bf16_grouped_qkv_plan_misses\":"
                  << grouped_qkv_stats.plan_misses
                  << ",\"bf16_grouped_qkv_dispatches\":"
                  << grouped_qkv_stats.dispatches
                  << ",\"bf16_grouped_qkv_retained_query_key_dispatches\":"
                  << grouped_qkv_stats.retained_query_key_dispatches
                  << ",\"bf16_grouped_qkv_kernel_setup_ms\":"
                  << grouped_qkv_stats.kernel_setup_ms
                  << ",\"bf16_grouped_qkv_argument_setup_ms\":"
                  << grouped_qkv_stats.argument_setup_ms
                  << ",\"bf16_grouped_gate_up_registered_entries\":"
                  << grouped_gate_up_stats.registered_entries
                  << ",\"bf16_grouped_gate_up_algorithm_entries\":"
                  << grouped_gate_up_stats.algorithm_entries
                  << ",\"bf16_grouped_gate_up_algorithm_hits\":"
                  << grouped_gate_up_stats.algorithm_hits
                  << ",\"bf16_grouped_gate_up_algorithm_misses\":"
                  << grouped_gate_up_stats.algorithm_misses
                  << ",\"bf16_grouped_gate_up_kernel_entries\":"
                  << grouped_gate_up_stats.kernel_entries
                  << ",\"bf16_grouped_gate_up_kernel_hits\":"
                  << grouped_gate_up_stats.kernel_hits
                  << ",\"bf16_grouped_gate_up_kernel_misses\":"
                  << grouped_gate_up_stats.kernel_misses
                  << ",\"bf16_grouped_gate_up_plan_entries\":"
                  << grouped_gate_up_stats.plan_entries
                  << ",\"bf16_grouped_gate_up_plan_hits\":"
                  << grouped_gate_up_stats.plan_hits
                  << ",\"bf16_grouped_gate_up_plan_misses\":"
                  << grouped_gate_up_stats.plan_misses
                  << ",\"bf16_grouped_gate_up_dispatches\":"
                  << grouped_gate_up_stats.dispatches
                  << ",\"bf16_grouped_gate_up_kernel_setup_ms\":"
                  << grouped_gate_up_stats.kernel_setup_ms
                  << ",\"bf16_grouped_gate_up_argument_setup_ms\":"
                  << grouped_gate_up_stats.argument_setup_ms
                  << ",\"fp32_attention_qk_solution_index\":"
                  << command.fp32_attention_qk_solution_index
                  << ",\"fp32_attention_pv_solution_index\":"
                  << command.fp32_attention_pv_solution_index
                  << ",\"fp32_solution_registered_entries\":"
                  << fp32_solution_stats.registered_entries
                  << ",\"fp32_solution_cached_algorithms\":"
                  << fp32_solution_stats.cached_algorithms
                  << ",\"fp32_solution_registry_hits\":"
                  << fp32_solution_stats.registry_hits
                  << ",\"fp32_solution_registry_misses\":"
                  << fp32_solution_stats.registry_misses
                  << ",\"fp32_solution_cache_hits\":"
                  << fp32_solution_stats.cache_hits
                  << ",\"fp32_solution_cache_misses\":"
                  << fp32_solution_stats.cache_misses
                  << ",\"fp32_solution_dispatches\":"
                  << fp32_solution_stats.dispatches
                  << ",\"device_total_bytes\":" << info.total_memory
                  << ",\"hip_runtime_version\":"
                  << microllm::runtime::hip_runtime_version()
                  << ",\"hip_driver_version\":" << microllm::runtime::hip_driver_version()
                  << ",\"compute_dtype\":\""
                  << (command.fp8_linear
                          ? fp8_compute_dtype(command)
                          : command.bf16_attention
                          ? "float32_with_bf16_ffn_attention"
                          : command.bf16_ffn ? "float32_with_bf16_ffn" : "float32")
                  << "\""
                  << ",\"inference_weight_policy\":\""
                  << (command.fp8_linear
                          ? fp8_storage_policy(command)
                          : command.bf16_attention
                          ? "single_representation_bf16_ffn_attention"
                          : command.bf16_ffn ? "single_representation_bf16_ffn" : "float32")
                  << "\""
                  << ",\"workload\":\"" << command.workload << "\""
                  << ",\"bf16_ffn_converted_tensors\":"
                  << bf16_report.converted_tensors
                  << ",\"bf16_ffn_arena_enabled\":"
                  << (model.bf16_ffn_arena_enabled() ? "true" : "false")
                  << ",\"bf16_ffn_norm_fusion_enabled\":"
                  << (model.bf16_ffn_norm_fusion_enabled() ? "true" : "false")
                  << ",\"bf16_ffn_arena_entries\":"
                  << bf16_arena_stats.entries
                  << ",\"bf16_ffn_arena_hits\":"
                  << bf16_arena_stats.hits
                  << ",\"bf16_ffn_arena_misses\":"
                  << bf16_arena_stats.misses
                  << ",\"bf16_ffn_arena_eligible_calls\":"
                  << bf16_arena_stats.eligible_calls
                  << ",\"bf16_ffn_arena_bypassed_calls\":"
                  << bf16_arena_stats.bypassed_calls
                  << ",\"bf16_ffn_arena_minimum_rows\":"
                  << bf16_arena_stats.minimum_rows
                  << ",\"bf16_ffn_arena_capacity_bytes\":"
                  << bf16_arena_stats.capacity_bytes
                  << ",\"bf16_qkv_arena_enabled\":"
                  << (model.bf16_qkv_arena_enabled() ? "true" : "false")
                  << ",\"bf16_attention_norm_fusion_enabled\":"
                  << (model.bf16_attention_norm_fusion_enabled() ? "true" : "false")
                  << ",\"bf16_qkv_arena_entries\":"
                  << bf16_qkv_arena_stats.entries
                  << ",\"bf16_qkv_arena_hits\":"
                  << bf16_qkv_arena_stats.hits
                  << ",\"bf16_qkv_arena_misses\":"
                  << bf16_qkv_arena_stats.misses
                  << ",\"bf16_qkv_arena_eligible_calls\":"
                  << bf16_qkv_arena_stats.eligible_calls
                  << ",\"bf16_qkv_arena_bypassed_calls\":"
                  << bf16_qkv_arena_stats.bypassed_calls
                  << ",\"bf16_qkv_arena_minimum_rows\":"
                  << bf16_qkv_arena_stats.minimum_rows
                  << ",\"bf16_qkv_arena_capacity_bytes\":"
                  << bf16_qkv_arena_stats.capacity_bytes
                  << ",\"attention_core_arena_enabled\":"
                  << (model.attention_core_arena_enabled() ? "true" : "false")
                  << ",\"attention_core_arena_entries\":"
                  << attention_core_arena_stats.entries
                  << ",\"attention_core_arena_hits\":"
                  << attention_core_arena_stats.hits
                  << ",\"attention_core_arena_misses\":"
                  << attention_core_arena_stats.misses
                  << ",\"attention_core_arena_eligible_calls\":"
                  << attention_core_arena_stats.eligible_calls
                  << ",\"attention_core_arena_bypassed_calls\":"
                  << attention_core_arena_stats.bypassed_calls
                  << ",\"attention_core_arena_minimum_sequence\":"
                  << attention_core_arena_stats.minimum_sequence
                  << ",\"attention_core_arena_capacity_bytes\":"
                  << attention_core_arena_stats.capacity_bytes
                  << ",\"cached_attention_splits\":"
                  << model.cached_attention_split_sequence_splits()
                  << ",\"cached_attention_minimum_sequence\":"
                  << model.cached_attention_split_minimum_sequence()
                  << ",\"bf16_attention_converted_tensors\":"
                  << bf16_attention_report.converted_tensors
                  << ",\"fp8_converted_tensors\":"
                  << fp8_report.converted_tensors
                  << ",\"fp8_linears_covered\":"
                  << fp8_report.linears_covered
                  << ",\"fp8_native_shapes\":"
                  << microllm::ops::fp8_dispatch_stats().native_shapes
                  << ",\"fp8_software_fallback_shapes\":"
                  << microllm::ops::fp8_dispatch_stats().software_fallback_shapes
                  << ",\"fp8_software_fallback_calls\":"
                  << microllm::ops::fp8_dispatch_stats().software_fallback_calls
                  << ",\"fp8_outer_row_fallback_calls\":"
                  << microllm::ops::fp8_dispatch_stats().outer_row_fallback_calls
                  << ",\"fp8_outer_row_native_status\":"
                  << microllm::ops::fp8_dispatch_stats().outer_row_native_status
                  << ",\"fp8_output_column_scale_calls\":"
                  << microllm::ops::fp8_dispatch_stats().output_column_scale_calls
                  << ",\"fp8_output_column_native_status\":"
                  << microllm::ops::fp8_dispatch_stats().output_column_native_status
                  << ",\"fp8_dynamic_tensor_calls\":"
                  << microllm::ops::fp8_dynamic_quant_stats().tensor_calls
                  << ",\"fp8_dynamic_row_calls\":"
                  << microllm::ops::fp8_dynamic_quant_stats().row_calls
                  << ",\"fp8_dynamic_tensor_elements\":"
                  << microllm::ops::fp8_dynamic_quant_stats().tensor_elements
                  << ",\"fp8_dynamic_row_elements\":"
                  << microllm::ops::fp8_dynamic_quant_stats().row_elements
                  << ",\"fp8_dynamic_column_calls\":"
                  << microllm::ops::fp8_dynamic_quant_stats().column_calls
                  << ",\"fp8_dynamic_column_elements\":"
                  << microllm::ops::fp8_dynamic_quant_stats().column_elements
                  << ",\"fp8_dynamic_clipped_tensor_calls\":"
                  << microllm::ops::fp8_dynamic_quant_stats().clipped_tensor_calls
                  << ",\"fp8_activation_scale\":"
                  << command.fp8_activation_scale
                  << ",\"fp8_activation_minimum_scale\":"
                  << command.fp8_activation_minimum_scale
                  << ",\"fp8_weight_scale\":"
                  << command.fp8_weight_scale
                  << ",\"fp8_weight_scale_mode\":\""
                  << command.fp8_weight_scale_mode << "\""
                  << ",\"fp8_weight_scale_scope\":\""
                  << command.fp8_weight_scale_scope << "\""
                  << ",\"fp8_activation_scale_mode\":\""
                  << command.fp8_activation_scale_mode << "\""
                  << ",\"fp8_diagnostic_mode\":\""
                  << command.fp8_diagnostic_mode << "\""
                  << ",\"fp8_fp32_layers\":\""
                  << command.fp8_fp32_layers << "\""
                  << ",\"fp8_weight_scale_min\":"
                  << fp8_report.minimum_weight_scale
                  << ",\"fp8_weight_scale_max\":"
                  << fp8_report.maximum_weight_scale
                  << ",\"fp8_weight_bytes_scanned\":"
                  << fp8_report.weight_bytes_scanned
                  << ",\"fp8_device_weight_bytes_scanned\":"
                  << fp8_report.device_weight_bytes_scanned
                  << ",\"fp8_device_amax_tensors\":"
                  << fp8_report.device_amax_tensors
                  << ",\"fp8_host_scale_summary_available\":"
                  << (fp8_report.host_scale_summary_available ? "true" : "false")
                  << ",\"fp32_weight_bytes_released\":"
                  << bf16_report.fp32_bytes_released
                  + bf16_attention_report.fp32_bytes_released
                  + fp8_report.fp32_bytes_released
                  << ",\"bf16_weight_bytes_retained\":"
                  << bf16_report.bf16_bytes_retained
                  + bf16_attention_report.bf16_bytes_retained
                  << ",\"fp8_weight_bytes_retained\":"
                  << fp8_report.fp8_bytes_retained
                  << ",\"fp8_scale_bytes_retained\":"
                  << fp8_report.scale_bytes_retained
                  << ",\"resident_weight_bytes\":"
                  << external.model.weight_bytes(sizeof(float)) -
                         bf16_report.fp32_bytes_released +
                         bf16_report.bf16_bytes_retained -
                         bf16_attention_report.fp32_bytes_released +
                         bf16_attention_report.bf16_bytes_retained -
                         fp8_report.fp32_bytes_released +
                         fp8_report.fp8_bytes_retained +
                         fp8_report.scale_bytes_retained
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
                  << ",\"weight_preparation_ms\":"
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
        std::cout << ",\"allocation_source_diagnostics\":"
                  << (command.allocation_source_diagnostics ? "true" : "false")
                  << ",\"allocation_source_calls\":"
                  << allocation_sources.calls
                  << ",\"allocation_source_bytes\":"
                  << allocation_sources.bytes
                  << ",\"allocation_source_records\":[";
        for (std::size_t index = 0;
             index < allocation_sources.records.size(); ++index) {
            if (index != 0) std::cout << ',';
            const auto& record = allocation_sources.records[index];
            std::cout << "{\"source\":\""
                      << microllm::runtime::allocation_source_name(record.source)
                      << "\",\"device\":\"" << record.device.str()
                      << "\",\"allocation_bytes\":"
                      << record.allocation_bytes
                      << ",\"calls\":" << record.calls
                      << ",\"total_bytes\":" << record.total_bytes << '}';
        }
        std::cout << ']';
        std::cout << ",\"strided_copy_diagnostics\":"
                  << (command.strided_copy_diagnostics ? "true" : "false")
                  << ",\"strided_copy_calls\":" << strided_copies.calls
                  << ",\"strided_copy_elements\":"
                  << strided_copies.elements
                  << ",\"strided_copy_bytes\":" << strided_copies.bytes
                  << ",\"strided_copy_records\":[";
        for (std::size_t index = 0;
             index < strided_copies.records.size(); ++index) {
            if (index != 0) std::cout << ',';
            const auto& record = strided_copies.records[index];
            std::cout << "{\"source\":\""
                      << microllm::runtime::allocation_source_name(
                             record.source)
                      << "\",\"device\":\"" << record.device.str()
                      << "\",\"element_bytes\":" << record.element_bytes
                      << ",\"calls\":" << record.calls
                      << ",\"elements\":" << record.elements
                      << ",\"bytes\":" << record.bytes
                      << ",\"shape\":[";
            for (std::size_t dimension = 0;
                 dimension < record.shape.size(); ++dimension) {
                if (dimension != 0) std::cout << ',';
                std::cout << record.shape[dimension];
            }
            std::cout << "],\"strides\":[";
            for (std::size_t dimension = 0;
                 dimension < record.strides.size(); ++dimension) {
                if (dimension != 0) std::cout << ',';
                std::cout << record.strides[dimension];
            }
            std::cout << "]}";
        }
        std::cout << ']';
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
