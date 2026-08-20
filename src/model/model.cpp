#include <microllm/model/model.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <microllm/ops/ops.h>
#include <microllm/profiling/trace.h>
#include <microllm/runtime/runtime.h>

namespace microllm::model {
namespace {

using autograd::Value;

Tensor prepare_cached_sequence(Tensor& cached, const Tensor& current,
                               std::int64_t position, std::int64_t capacity) {
    if (current.dtype() != DType::Float32 || current.ndim() != 4) {
        throw std::invalid_argument("cached K/V tensors must be float32 rank four");
    }
    if (position < 0 || position >= capacity || current.shape()[0] != 1 ||
        current.shape()[2] != 1) {
        throw std::out_of_range("cached K/V position is outside the preallocated capacity");
    }
    const auto packed = current.is_contiguous() ? current : current.contiguous();
    const auto heads = current.shape()[1];
    const auto width = current.shape()[3];
    if (!cached.defined()) {
        if (position != 0) throw std::invalid_argument("KV cache must start at position zero");
        Tensor backing({1, heads, capacity, width}, current.dtype(), current.device());
        cached = Tensor::from_storage(backing.storage(), {1, heads, 1, width},
                                      backing.strides(), 0, current.dtype());
    } else if (cached.ndim() != 4 || cached.shape()[0] != 1 ||
               cached.shape()[1] != heads || cached.shape()[2] != position ||
               cached.shape()[3] != width || cached.device() != current.device()) {
        throw std::invalid_argument("cached and current K/V shapes are incompatible");
    } else {
        cached = Tensor::from_storage(cached.storage(), {1, heads, position + 1, width},
                                      cached.strides(), cached.storage_offset(), cached.dtype());
    }
    return packed;
}

Tensor random_tensor(Shape shape, std::mt19937_64& generator, float standard_deviation) {
    std::normal_distribution<float> distribution(0.0F, standard_deviation);
    std::vector<float> values(static_cast<std::size_t>(checked_numel(shape)));
    for (auto& value : values) value = distribution(generator);
    return Tensor::from_vector(values, std::move(shape));
}

template <typename Predicate>
Bf16WeightPreparationReport prepare_bf16_weights(
    const NamedValues& named, std::size_t expected_count, Device device,
    Predicate&& selected) {
    struct Prepared {
        Value* parameter = nullptr;
        Tensor bf16;
    };
    std::vector<Prepared> prepared;
    Bf16WeightPreparationReport report;
    for (const auto& [name, parameter] : named) {
        if (!selected(name)) continue;
        if (parameter->data().dtype() != DType::Float32 ||
            !parameter->data().is_contiguous()) {
            throw std::logic_error(
                "BF16 inference preparation requires contiguous FP32 source weights");
        }
        const auto elements = static_cast<std::uint64_t>(parameter->data().numel());
        prepared.push_back({parameter, ops::cast(parameter->data(), DType::BFloat16)});
        ++report.converted_tensors;
        report.fp32_bytes_released += elements * sizeof(float);
        report.bf16_bytes_retained += elements * sizeof(std::uint16_t);
    }
    if (report.converted_tensors != expected_count) {
        throw std::logic_error("model exposed an unexpected BF16 inference weight count");
    }
    runtime::synchronize(device);
    for (auto& item : prepared) {
        *item.parameter = Value(std::move(item.bf16), false);
    }
    return report;
}

Value parameter(Shape shape, std::mt19937_64& generator, float standard_deviation) {
    return Value(random_tensor(std::move(shape), generator, standard_deviation), true);
}

class Linear {
public:
    Linear(std::int64_t input, std::int64_t output, std::mt19937_64& generator,
           const ModelConfig& config, bool with_bias = false)
        : weight_(parameter({input, output}, generator,
                            1.0F / std::sqrt(static_cast<float>(input)))),
          precision_(config.linear_precision),
          activation_scale_(config.fp8_activation_scale),
          weight_scale_(config.fp8_weight_scale), has_bias_(with_bias) {
        if (has_bias_) {
            bias_ = Value(Tensor({output}), true);
            bias_.mutable_data().fill(0.0F);
        }
    }

    Value forward_without_bias(const Value& input) {
        if (precision_ == LinearPrecision::Float8E4M3FNUZ) {
            return autograd::fp8_matmul(input, weight_, activation_scale_, weight_scale_);
        }
        return autograd::matmul(input, weight_);
    }
    Value forward(const Value& input) {
        auto output = forward_without_bias(input);
        return has_bias_ ? autograd::add_bias(output, bias_) : output;
    }
    Tensor forward_tensor_without_bias(const Tensor& input) {
        if (precision_ == LinearPrecision::Float8E4M3FNUZ) {
            return ops::fp8_matmul(
                ops::quantize_fp8(input, DType::Float8E4M3FNUZ, activation_scale_),
                ops::quantize_fp8(weight_.data(), DType::Float8E4M3FNUZ, weight_scale_),
                DType::Float32);
        }
        if (weight_.data().dtype() == DType::BFloat16) {
            return ops::bf16_matmul(input, weight_.data());
        }
        return ops::matmul_with_implementation(input, weight_.data(),
                                               ops::MatmulImplementation::Auto);
    }
    Tensor forward_tensor(const Tensor& input) {
        auto output = forward_tensor_without_bias(input);
        return has_bias_ ? ops::add_bias(output, bias_.data()) : output;
    }
    Value& weight() noexcept { return weight_; }
    [[nodiscard]] const Tensor& weight_data() const noexcept { return weight_.data(); }
    [[nodiscard]] bool has_bias() const noexcept { return has_bias_; }
    Value& bias() noexcept { return bias_; }

private:
    Value weight_;
    LinearPrecision precision_ = LinearPrecision::Float32;
    float activation_scale_ = 1.0F;
    float weight_scale_ = 1.0F;
    bool has_bias_ = false;
    Value bias_;
};

class Norm {
public:
    explicit Norm(std::int64_t dimension, float epsilon = 1.0e-5F)
        : weight_(Tensor({dimension}), true), epsilon_(epsilon) {
        weight_.mutable_data().fill(1.0F);
    }

    Value forward(const Value& input) { return autograd::rms_norm(input, weight_, epsilon_); }
    Tensor forward_tensor(const Tensor& input) {
        return ops::rms_norm(input, weight_.data(), epsilon_);
    }
    ops::TensorPair add_forward_tensor(const Tensor& left, const Tensor& right) {
        return ops::add_rms_norm(left, right, weight_.data(), epsilon_);
    }
    Value& weight() noexcept { return weight_; }

private:
    Value weight_;
    float epsilon_ = 1.0e-5F;
};

class Attention {
public:
    Attention(const ModelConfig& config, std::mt19937_64& generator)
        : config_(config),
          query_(config.dimension, config.dimension, generator, config, config.attention_bias),
          key_(config.dimension, config.kv_dimension(), generator, config, config.attention_bias),
          value_(config.dimension, config.kv_dimension(), generator, config, config.attention_bias),
          output_(config.dimension, config.dimension, generator, config) {}

    Value forward(const Value& input) {
        if (input.data().ndim() != 3) throw std::invalid_argument("attention input must be BxTxD");
        const auto batch = input.data().shape()[0];
        const auto sequence = input.data().shape()[1];
        const auto flat = autograd::reshape(input, {batch * sequence, config_.dimension});
        const auto fuse_query_bias = config_.rope_layout == RopeLayout::SplitHalf &&
                                     query_.has_bias();
        const auto fuse_key_bias = config_.rope_layout == RopeLayout::SplitHalf &&
                                   key_.has_bias();
        auto query = autograd::reshape(
            fuse_query_bias ? query_.forward_without_bias(flat) : query_.forward(flat),
                                       {batch, sequence, config_.heads, config_.head_dimension()});
        auto key = autograd::reshape(
            fuse_key_bias ? key_.forward_without_bias(flat) : key_.forward(flat),
                                     {batch, sequence, config_.kv_heads, config_.head_dimension()});
        auto value = autograd::reshape(value_.forward(flat),
                                       {batch, sequence, config_.kv_heads, config_.head_dimension()});
        const auto transposed_query = autograd::transpose(query, 1, 2);
        const auto transposed_key = autograd::transpose(key, 1, 2);
        if (config_.rope_layout == RopeLayout::SplitHalf) {
            query = fuse_query_bias
                        ? autograd::rope_split_half_bias(
                              transposed_query, query_.bias(), 0, config_.rope_base)
                        : autograd::rope_split_half(
                              transposed_query, 2, 0, config_.rope_base);
            key = fuse_key_bias
                      ? autograd::rope_split_half_bias(
                            transposed_key, key_.bias(), 0, config_.rope_base)
                      : autograd::rope_split_half(
                            transposed_key, 2, 0, config_.rope_base);
        } else {
            query = autograd::rope(transposed_query, 2, 0, config_.rope_base);
            key = autograd::rope(transposed_key, 2, 0, config_.rope_base);
        }
        value = autograd::transpose(value, 1, 2);
        if (config_.kv_heads != config_.heads) {
            const auto repeats = config_.heads / config_.kv_heads;
            key = autograd::repeat_interleave(key, 1, repeats);
            value = autograd::repeat_interleave(value, 1, repeats);
        }
        const auto key_transposed = autograd::transpose(key, -2, -1);
        const auto scores = autograd::scale(
            autograd::matmul(query, key_transposed),
            1.0F / std::sqrt(static_cast<float>(config_.head_dimension())));
        const auto probabilities = autograd::causal_softmax(scores);
        auto context = autograd::matmul(probabilities, value);
        context = autograd::contiguous(autograd::transpose(context, 1, 2));
        context = autograd::reshape(context, {batch * sequence, config_.dimension});
        return autograd::reshape(output_.forward(context),
                                 {batch, sequence, config_.dimension});
    }

    Tensor forward_tensor(const Tensor& input) {
        if (input.ndim() != 3) throw std::invalid_argument("attention input must be BxTxD");
        const auto batch = input.shape()[0];
        const auto sequence = input.shape()[1];
        const auto flat = input.reshape({batch * sequence, config_.dimension});
        Tensor query_projection;
        Tensor key_projection;
        Tensor value_projection;
        if (query_.weight_data().dtype() == DType::BFloat16) {
            const auto projections = ops::bf16_qkv_projection(
                flat, query_.weight_data(), key_.weight_data(), value_.weight_data());
            query_projection = query_.has_bias()
                                   ? ops::add_bias(projections.first, query_.bias().data())
                                   : projections.first;
            key_projection = key_.has_bias()
                                 ? ops::add_bias(projections.second, key_.bias().data())
                                 : projections.second;
            value_projection = value_.has_bias()
                                   ? ops::add_bias(projections.third, value_.bias().data())
                                   : projections.third;
        } else {
            query_projection = query_.forward_tensor(flat);
            key_projection = key_.forward_tensor(flat);
            value_projection = value_.forward_tensor(flat);
        }
        auto query = query_projection
                         .reshape({batch, sequence, config_.heads, config_.head_dimension()})
                         .transpose(1, 2)
                         .contiguous();
        auto key = key_projection
                       .reshape({batch, sequence, config_.kv_heads,
                                 config_.head_dimension()})
                       .transpose(1, 2)
                       .contiguous();
        auto value = value_projection
                         .reshape({batch, sequence, config_.kv_heads,
                                   config_.head_dimension()})
                         .transpose(1, 2)
                         .contiguous();
        if (config_.rope_layout == RopeLayout::SplitHalf) {
            query = ops::rope_split_half(query, 2, 0, config_.rope_base);
            key = ops::rope_split_half(key, 2, 0, config_.rope_base);
        } else {
            query = ops::rope(query, 2, 0, config_.rope_base);
            key = ops::rope(key, 2, 0, config_.rope_base);
        }
        if (config_.kv_heads != config_.heads) {
            const auto repeats = config_.heads / config_.kv_heads;
            key = ops::repeat_interleave(key, 1, repeats);
            value = ops::repeat_interleave(value, 1, repeats);
        }
        const auto scores = ops::scale(
            ops::matmul(query, key.transpose(-2, -1).contiguous()),
            1.0F / std::sqrt(static_cast<float>(config_.head_dimension())));
        const auto probabilities = ops::causal_softmax(scores);
        auto context = ops::matmul(probabilities, value)
                           .transpose(1, 2)
                           .contiguous()
                           .reshape({batch * sequence, config_.dimension});
        return output_.forward_tensor(context).reshape(
            {batch, sequence, config_.dimension});
    }

    Tensor forward_cached(const Tensor& input, inference::KVCache::LayerState& cache,
                          std::int64_t position, std::int64_t cache_capacity) {
        if (input.shape().size() != 3 || input.shape()[0] != 1 || input.shape()[1] != 1) {
            throw std::invalid_argument("cached attention expects one B=1 token");
        }
        const auto flat = input.reshape({1, config_.dimension});
        const auto fuse_query_bias = config_.rope_layout == RopeLayout::SplitHalf &&
                                     query_.has_bias();
        const auto fuse_key_bias = config_.rope_layout == RopeLayout::SplitHalf &&
                                   key_.has_bias();
        Tensor query_projection;
        Tensor key_projection;
        Tensor value_projection;
        if (query_.weight_data().dtype() == DType::BFloat16) {
            const auto projections = ops::bf16_qkv_projection(
                flat, query_.weight_data(), key_.weight_data(), value_.weight_data());
            query_projection = fuse_query_bias
                                   ? projections.first
                                   : query_.has_bias()
                                         ? ops::add_bias(projections.first, query_.bias().data())
                                         : projections.first;
            key_projection = fuse_key_bias
                                 ? projections.second
                                 : key_.has_bias()
                                       ? ops::add_bias(projections.second, key_.bias().data())
                                       : projections.second;
            value_projection = value_.has_bias()
                                   ? ops::add_bias(projections.third, value_.bias().data())
                                   : projections.third;
        } else {
            query_projection = fuse_query_bias ? query_.forward_tensor_without_bias(flat)
                                                : query_.forward_tensor(flat);
            key_projection = fuse_key_bias ? key_.forward_tensor_without_bias(flat)
                                            : key_.forward_tensor(flat);
            value_projection = value_.forward_tensor(flat);
        }
        auto query = query_projection
                         .reshape({1, 1, config_.heads, config_.head_dimension()})
                         .transpose(1, 2);
        auto key = key_projection
                       .reshape({1, 1, config_.kv_heads, config_.head_dimension()})
                       .transpose(1, 2);
        auto value = value_projection
                         .reshape({1, 1, config_.kv_heads, config_.head_dimension()})
                         .transpose(1, 2);
        if (config_.rope_layout == RopeLayout::SplitHalf) {
            query = fuse_query_bias
                        ? ops::rope_split_half_bias(
                              query, query_.bias().data(), position, config_.rope_base)
                        : ops::rope_split_half(query, 2, position, config_.rope_base);
            key = fuse_key_bias
                      ? ops::rope_split_half_bias(
                            key, key_.bias().data(), position, config_.rope_base)
                      : ops::rope_split_half(key, 2, position, config_.rope_base);
        } else {
            query = ops::rope(query, 2, position, config_.rope_base);
            key = ops::rope(key, 2, position, config_.rope_base);
        }
        const auto packed_key =
            prepare_cached_sequence(cache.key, key, position, cache_capacity);
        const auto packed_value =
            prepare_cached_sequence(cache.value, value, position, cache_capacity);
        ops::kv_cache_store_pair_(cache.key, cache.value, packed_key, packed_value,
                                  position);
        const auto repeats = config_.heads / config_.kv_heads;
        auto context = ops::cached_gqa_attention(
                           query, cache.key, cache.value, repeats,
                           1.0F / std::sqrt(static_cast<float>(config_.head_dimension())))
                           .transpose(1, 2)
                           .contiguous()
                           .reshape({1, config_.dimension});
        return output_.forward_tensor(context).reshape({1, 1, config_.dimension});
    }

    void append_named(const std::string& prefix, NamedValues& values) {
        values.emplace_back(prefix + ".q_proj.weight", &query_.weight());
        if (query_.has_bias()) values.emplace_back(prefix + ".q_proj.bias", &query_.bias());
        values.emplace_back(prefix + ".k_proj.weight", &key_.weight());
        if (key_.has_bias()) values.emplace_back(prefix + ".k_proj.bias", &key_.bias());
        values.emplace_back(prefix + ".v_proj.weight", &value_.weight());
        if (value_.has_bias()) values.emplace_back(prefix + ".v_proj.bias", &value_.bias());
        values.emplace_back(prefix + ".o_proj.weight", &output_.weight());
    }

private:
    ModelConfig config_;
    Linear query_;
    Linear key_;
    Linear value_;
    Linear output_;
};

class FeedForward {
public:
    FeedForward(const ModelConfig& config, std::mt19937_64& generator)
        : config_(config),
          gate_(config.dimension, config.ffn_dimension, generator, config),
          up_(config.dimension, config.ffn_dimension, generator, config),
          down_(config.ffn_dimension, config.dimension, generator, config) {}

    Value forward(const Value& input) {
        const auto batch = input.data().shape()[0];
        const auto sequence = input.data().shape()[1];
        const auto flat = autograd::reshape(input, {batch * sequence, config_.dimension});
        // Keep projection evaluation order explicit so traces and failure attribution are stable.
        const auto gate = gate_.forward(flat);
        const auto up = up_.forward(flat);
        const auto activated = autograd::swiglu(gate, up);
        return autograd::reshape(down_.forward(activated),
                                 {batch, sequence, config_.dimension});
    }

    Tensor forward_tensor(const Tensor& input) {
        if (input.ndim() != 3) throw std::invalid_argument("FFN input must be BxTxD");
        const auto batch = input.shape()[0];
        const auto sequence = input.shape()[1];
        const auto flat = input.reshape({batch * sequence, config_.dimension});
        Tensor output;
        if (gate_.weight_data().dtype() == DType::BFloat16) {
            if (up_.weight_data().dtype() != DType::BFloat16 ||
                down_.weight_data().dtype() != DType::BFloat16) {
                throw std::logic_error("FFN inference weights have mixed preparation state");
            }
            output = ops::bf16_ffn(flat, gate_.weight_data(), up_.weight_data(),
                                   down_.weight_data());
        } else {
            const auto activated = ops::swiglu(gate_.forward_tensor(flat),
                                                up_.forward_tensor(flat));
            output = down_.forward_tensor(activated);
        }
        return output.reshape({batch, sequence, config_.dimension});
    }


    void append_named(const std::string& prefix, NamedValues& values) {
        values.emplace_back(prefix + ".gate_proj.weight", &gate_.weight());
        values.emplace_back(prefix + ".up_proj.weight", &up_.weight());
        values.emplace_back(prefix + ".down_proj.weight", &down_.weight());
    }

private:
    ModelConfig config_;
    Linear gate_;
    Linear up_;
    Linear down_;
};

class Block {
public:
    Block(const ModelConfig& config, std::mt19937_64& generator)
        : attention_norm_(config.dimension, config.rms_norm_epsilon),
          attention_(config, generator),
          ffn_norm_(config.dimension, config.rms_norm_epsilon),
          feed_forward_(config, generator) {}

    Value forward(const Value& input) {
        auto hidden = autograd::add(input, attention_.forward(attention_norm_.forward(input)));
        return autograd::add(hidden, feed_forward_.forward(ffn_norm_.forward(hidden)));
    }

    Tensor forward_tensor(const Tensor& input) {
        auto hidden = ops::add(input, attention_.forward_tensor(
                                          attention_norm_.forward_tensor(input)));
        return ops::add(hidden, feed_forward_.forward_tensor(
                                    ffn_norm_.forward_tensor(hidden)));
    }

    Tensor forward_cached(const Tensor& input, inference::KVCache::LayerState& cache,
                          std::int64_t position, std::int64_t cache_capacity) {
        auto attention = attention_.forward_cached(attention_norm_.forward_tensor(input), cache,
                                                   position, cache_capacity);
        auto residual_and_norm = ffn_norm_.add_forward_tensor(input, attention);
        return ops::add(residual_and_norm.first,
                        feed_forward_.forward_tensor(residual_and_norm.second));
    }

    void append_named(const std::string& prefix, NamedValues& values) {
        values.emplace_back(prefix + ".attention_norm.weight", &attention_norm_.weight());
        attention_.append_named(prefix + ".attention", values);
        values.emplace_back(prefix + ".ffn_norm.weight", &ffn_norm_.weight());
        feed_forward_.append_named(prefix + ".feed_forward", values);
    }


private:
    Norm attention_norm_;
    Attention attention_;
    Norm ffn_norm_;
    FeedForward feed_forward_;
};

}  // namespace

struct TransformerModel::Impl {
    Impl(ModelConfig model_config, std::uint64_t seed)
        : config(std::move(model_config)),
          generator(seed),
          token_embedding(parameter({config.vocabulary_size, config.dimension}, generator, 0.02F)),
          final_norm(config.dimension, config.rms_norm_epsilon) {
        config.validate();
        blocks.reserve(static_cast<std::size_t>(config.layers));
        for (std::int64_t layer = 0; layer < config.layers; ++layer) {
            blocks.push_back(std::make_unique<Block>(config, generator));
        }
        if (!config.tie_embeddings) {
            output_head = std::make_unique<Linear>(config.dimension, config.vocabulary_size,
                                                   generator, config);
        }
    }

    ModelConfig config;
    std::mt19937_64 generator;
    Value token_embedding;
    std::vector<std::unique_ptr<Block>> blocks;
    Norm final_norm;
    std::unique_ptr<Linear> output_head;
    bool bf16_ffn_prepared = false;
    bool bf16_attention_prepared = false;
};

TransformerModel::TransformerModel(ModelConfig config, std::uint64_t seed)
    : impl_(std::make_unique<Impl>(std::move(config), seed)) {
    if (parameter_count() != impl_->config.parameter_count()) {
        throw std::logic_error("constructed model parameter count does not match ModelConfig");
    }
}
TransformerModel::~TransformerModel() = default;
TransformerModel::TransformerModel(TransformerModel&&) noexcept = default;
TransformerModel& TransformerModel::operator=(TransformerModel&&) noexcept = default;
const ModelConfig& TransformerModel::config() const noexcept { return impl_->config; }

Device TransformerModel::device() {
    return impl_->token_embedding.data().device();
}

void TransformerModel::to(Device target) {
    for (auto* value : parameters()) {
        value->mutable_data() = value->data().to(target);
        value->zero_grad();
    }
}

Value TransformerModel::forward(const Tensor& token_ids) {
    if (impl_->bf16_ffn_prepared || impl_->bf16_attention_prepared) {
        throw std::logic_error(
            "autograd forward is unavailable after BF16 FFN inference preparation; "
            "use forward_inference or forward_cached");
    }
    if (token_ids.dtype() != DType::Int32 || token_ids.ndim() != 2) {
        throw std::invalid_argument("model token IDs must be an int32 BxT tensor");
    }
    if (token_ids.shape()[1] > impl_->config.max_sequence_length) {
        throw std::invalid_argument("token sequence exceeds configured maximum");
    }
    const auto model_tokens = token_ids.device() == device() ? token_ids : token_ids.to(device());
    auto* trace = profiling::TraceSession::current();
    if (trace != nullptr) trace->record(profiling::TraceKind::Input, "model.tokens", model_tokens);
    profiling::TraceTimer model_timer(profiling::TraceKind::Model, "model.forward", device());

    profiling::TraceTimer embedding_timer(profiling::TraceKind::Layer,
                                           "model.embedding", device());
    auto hidden = autograd::embedding(impl_->token_embedding, model_tokens);
    embedding_timer.finish(hidden.data());
    for (std::size_t layer = 0; layer < impl_->blocks.size(); ++layer) {
        profiling::TraceTimer block_timer(
            profiling::TraceKind::Layer,
            "model.blocks." + std::to_string(layer), device());
        hidden = impl_->blocks[layer]->forward(hidden);
        block_timer.finish(hidden.data());
    }
    profiling::TraceTimer norm_timer(profiling::TraceKind::Layer,
                                      "model.final_norm", device());
    hidden = impl_->final_norm.forward(hidden);
    norm_timer.finish(hidden.data());
    const auto batch = token_ids.shape()[0];
    const auto sequence = token_ids.shape()[1];
    const auto flat = autograd::reshape(hidden, {batch * sequence, impl_->config.dimension});
    Value logits;
    if (impl_->config.tie_embeddings) {
        logits = autograd::matmul(flat, impl_->token_embedding, false, true);
    } else {
        logits = impl_->output_head->forward(flat);
    }
    auto output = autograd::reshape(logits, {batch, sequence, impl_->config.vocabulary_size});
    if (trace != nullptr) trace->record(profiling::TraceKind::Output, "model.logits", output.data());
    model_timer.finish(output.data());
    return output;
}

Tensor TransformerModel::forward_inference(const Tensor& token_ids) {
    if (token_ids.dtype() != DType::Int32 || token_ids.ndim() != 2) {
        throw std::invalid_argument("model token IDs must be an int32 BxT tensor");
    }
    if (token_ids.shape()[1] > impl_->config.max_sequence_length) {
        throw std::invalid_argument("token sequence exceeds configured maximum");
    }
    const auto model_tokens = token_ids.device() == device() ? token_ids : token_ids.to(device());
    auto hidden = ops::embedding(impl_->token_embedding.data(), model_tokens);
    for (auto& block : impl_->blocks) hidden = block->forward_tensor(hidden);
    hidden = impl_->final_norm.forward_tensor(hidden);
    const auto batch = token_ids.shape()[0];
    const auto sequence = token_ids.shape()[1];
    const auto flat = hidden.reshape({batch * sequence, impl_->config.dimension});
    Tensor logits;
    if (impl_->config.tie_embeddings) {
        logits = ops::matmul_with_implementation(
            flat, impl_->token_embedding.data(), ops::MatmulImplementation::Auto,
            false, true);
    } else {
        logits = impl_->output_head->forward_tensor(flat);
    }
    return logits.reshape({batch, sequence, impl_->config.vocabulary_size});
}

Value TransformerModel::loss(const Tensor& token_ids, const Tensor& targets) {
    if (targets.shape() != token_ids.shape()) {
        throw std::invalid_argument("language-model targets must match token shape");
    }
    const auto model_targets = targets.device() == device() ? targets : targets.to(device());
    return autograd::cross_entropy(forward(token_ids), model_targets);
}

Tensor TransformerModel::forward_cached(const Tensor& token_id, inference::KVCache& cache) {
    if (token_id.dtype() != DType::Int32 || token_id.shape() != Shape{1, 1}) {
        throw std::invalid_argument("cached forward expects one int32 token with shape 1x1");
    }
    if (cache.layer_count() != impl_->blocks.size() ||
        cache.max_sequence_length() > impl_->config.max_sequence_length) {
        throw std::invalid_argument("KV cache does not match model configuration");
    }
    if (cache.position() >= cache.max_sequence_length()) {
        throw std::out_of_range("KV cache has reached maximum sequence length");
    }
    const auto model_device = device();
    const auto device_token = token_id.device() == model_device ? token_id : token_id.to(model_device);
    auto hidden = ops::embedding(impl_->token_embedding.data(), device_token);
    for (std::size_t layer = 0; layer < impl_->blocks.size(); ++layer) {
        hidden = impl_->blocks[layer]->forward_cached(hidden, cache.mutable_layer(layer),
                                                      cache.position(),
                                                      cache.max_sequence_length());
    }
    hidden = impl_->final_norm.forward_tensor(hidden);
    const auto flat = hidden.reshape({1, impl_->config.dimension});
    Tensor logits;
    if (impl_->config.tie_embeddings) {
        logits = ops::matmul_with_implementation(
            flat, impl_->token_embedding.data(), ops::MatmulImplementation::Auto,
            false, true);
    } else {
        logits = impl_->output_head->forward_tensor(flat);
    }
    cache.advance();
    return logits.reshape({1, 1, impl_->config.vocabulary_size});
}

NamedValues TransformerModel::named_parameters() {
    NamedValues values;
    values.emplace_back("token_embedding.weight", &impl_->token_embedding);
    for (std::size_t layer = 0; layer < impl_->blocks.size(); ++layer) {
        impl_->blocks[layer]->append_named("blocks." + std::to_string(layer), values);
    }
    values.emplace_back("final_norm.weight", &impl_->final_norm.weight());
    if (impl_->output_head) values.emplace_back("output_head.weight", &impl_->output_head->weight());
    return values;
}

std::vector<Value*> TransformerModel::parameters() {
    const auto named = named_parameters();
    std::vector<Value*> values;
    values.reserve(named.size());
    for (const auto& [name, value] : named) {
        (void)name;
        values.push_back(value);
    }
    return values;
}

std::uint64_t TransformerModel::parameter_count() {
    std::uint64_t count = 0;
    for (const auto* value : parameters()) count += static_cast<std::uint64_t>(value->data().numel());
    return count;
}

Bf16FfnPreparationReport TransformerModel::prepare_bf16_ffn_inference() {
    if (impl_->bf16_ffn_prepared) {
        throw std::logic_error("BF16 FFN inference preparation is one-way and already complete");
    }
    if (impl_->config.linear_precision != LinearPrecision::Float32) {
        throw std::logic_error("BF16 inference preparation requires FP32 Linear policy");
    }
    // Transactional helper keeps every FP32 source alive until all casts finish.
    const auto report = prepare_bf16_weights(
        named_parameters(), impl_->blocks.size() * 3U, device(),
        [](const std::string& name) {
            return name.find(".feed_forward.") != std::string::npos;
        });
    impl_->bf16_ffn_prepared = true;
    return report;
}

bool TransformerModel::bf16_ffn_inference_prepared() const noexcept {
    return impl_->bf16_ffn_prepared;
}

Bf16WeightPreparationReport TransformerModel::prepare_bf16_attention_inference() {
    if (impl_->bf16_attention_prepared) {
        throw std::logic_error(
            "BF16 Attention inference preparation is one-way and already complete");
    }
    if (impl_->config.linear_precision != LinearPrecision::Float32) {
        throw std::logic_error("BF16 inference preparation requires FP32 Linear policy");
    }
    const auto report = prepare_bf16_weights(
        named_parameters(), impl_->blocks.size() * 4U, device(),
        [](const std::string& name) {
            return name.find(".attention.") != std::string::npos &&
                   name.ends_with(".weight");
        });
    impl_->bf16_attention_prepared = true;
    return report;
}

bool TransformerModel::bf16_attention_inference_prepared() const noexcept {
    return impl_->bf16_attention_prepared;
}

io::StateDict TransformerModel::state_dict(Device target) {
    io::StateDict state;
    for (const auto& [name, parameter] : named_parameters()) {
        auto copy = Tensor::from_vector(parameter->data().to_vector(), parameter->data().shape());
        if (target != Device::cpu()) copy = copy.to(target);
        state.emplace(name, std::move(copy));
    }
    return state;
}

LoadWeightsReport TransformerModel::load_state_dict(const io::StateDict& state,
                                                     const LoadWeightsOptions& options) {
    if (impl_->bf16_ffn_prepared || impl_->bf16_attention_prepared) {
        throw std::logic_error(
            "load weights before one-way BF16 FFN inference preparation");
    }
    LoadWeightsReport report;
    const auto named = named_parameters();
    std::set<std::string> target_names;
    for (const auto& [name, parameter] : named) {
        (void)parameter;
        target_names.insert(name);
    }
    for (const auto& [target, source] : options.mapping) {
        (void)source;
        if (!target_names.contains(target)) {
            report.incompatible.push_back("mapping target is not a model parameter: " + target);
        }
    }

    struct Prepared {
        std::string name;
        Value* parameter = nullptr;
        Tensor tensor;
    };
    std::vector<Prepared> prepared;
    std::set<std::string> consumed;
    for (const auto& [target_name, parameter] : named) {
        const auto mapping = options.mapping.find(target_name);
        const auto source_name = mapping == options.mapping.end()
                                     ? target_name
                                     : mapping->second.name;
        const auto transform = mapping == options.mapping.end()
                                   ? WeightTransform::Identity
                                   : mapping->second.transform;
        const auto found = state.find(source_name);
        if (found == state.end()) {
            report.missing.push_back(target_name + " <- " + source_name);
            continue;
        }
        consumed.insert(source_name);
        auto source = found->second;
        if (!source.defined() || source.dtype() != DType::Float32) {
            report.incompatible.push_back(target_name + " requires a defined float32 source");
            continue;
        }
        if (transform == WeightTransform::Transpose2D) {
            if (source.ndim() != 2) {
                report.incompatible.push_back(target_name + " transpose requires rank two");
                continue;
            }
            source = source.transpose(0, 1).contiguous();
        }
        if (source.shape() != parameter->data().shape()) {
            std::ostringstream message;
            message << target_name << " shape mismatch: source=[";
            for (std::size_t index = 0; index < source.shape().size(); ++index) {
                if (index != 0) message << ',';
                message << source.shape()[index];
            }
            message << "] target=[";
            for (std::size_t index = 0; index < parameter->data().shape().size(); ++index) {
                if (index != 0) message << ',';
                message << parameter->data().shape()[index];
            }
            message << ']';
            report.incompatible.push_back(message.str());
            continue;
        }
        auto copy = Tensor::from_vector(source.to_vector(), source.shape());
        if (parameter->data().device() != Device::cpu()) {
            copy = copy.to(parameter->data().device());
        }
        prepared.push_back({target_name, parameter, std::move(copy)});
    }
    for (const auto& [name, tensor] : state) {
        (void)tensor;
        if (!consumed.contains(name)) report.unexpected.push_back(name);
    }

    if (options.strict && !report.complete()) {
        std::ostringstream message;
        message << "strict weight load failed";
        for (const auto& missing : report.missing) message << "\nmissing: " << missing;
        for (const auto& unexpected : report.unexpected) message << "\nunexpected: " << unexpected;
        for (const auto& incompatible : report.incompatible) {
            message << "\nincompatible: " << incompatible;
        }
        throw std::invalid_argument(message.str());
    }
    for (auto& item : prepared) {
        item.parameter->mutable_data() = std::move(item.tensor);
        item.parameter->zero_grad();
        report.loaded.push_back(std::move(item.name));
    }
    return report;
}

LoadWeightsReport TransformerModel::load_safetensors(
    const std::filesystem::path& path, const LoadWeightsOptions& options) {
    return load_state_dict(io::load_safetensors(path), options);
}

LoadWeightsReport TransformerModel::load_safetensors_files(
    const std::vector<std::filesystem::path>& paths,
    const LoadWeightsOptions& options) {
    return load_state_dict(io::load_safetensors_files(paths), options);
}

LoadWeightsReport TransformerModel::load_safetensors_index(
    const std::filesystem::path& index_path, const LoadWeightsOptions& options) {
    return load_state_dict(io::load_safetensors_index(index_path), options);
}

void TransformerModel::save_safetensors(
    const std::filesystem::path& path,
    const io::SafetensorsSaveOptions& options) {
    io::save_safetensors(path, state_dict(), options);
}

WeightMapping qwen_style_weight_mapping(const ModelConfig& config) {
    config.validate();
    WeightMapping mapping;
    mapping.emplace("token_embedding.weight",
                    WeightSource{"model.embed_tokens.weight", WeightTransform::Identity});
    for (std::int64_t layer = 0; layer < config.layers; ++layer) {
        const auto target = "blocks." + std::to_string(layer);
        const auto source = "model.layers." + std::to_string(layer);
        mapping.emplace(target + ".attention_norm.weight",
                        WeightSource{source + ".input_layernorm.weight",
                                     WeightTransform::Identity});
        mapping.emplace(target + ".attention.q_proj.weight",
                        WeightSource{source + ".self_attn.q_proj.weight",
                                     WeightTransform::Transpose2D});
        mapping.emplace(target + ".attention.k_proj.weight",
                        WeightSource{source + ".self_attn.k_proj.weight",
                                     WeightTransform::Transpose2D});
        mapping.emplace(target + ".attention.v_proj.weight",
                        WeightSource{source + ".self_attn.v_proj.weight",
                                     WeightTransform::Transpose2D});
        if (config.attention_bias) {
            mapping.emplace(target + ".attention.q_proj.bias",
                            WeightSource{source + ".self_attn.q_proj.bias",
                                         WeightTransform::Identity});
            mapping.emplace(target + ".attention.k_proj.bias",
                            WeightSource{source + ".self_attn.k_proj.bias",
                                         WeightTransform::Identity});
            mapping.emplace(target + ".attention.v_proj.bias",
                            WeightSource{source + ".self_attn.v_proj.bias",
                                         WeightTransform::Identity});
        }
        mapping.emplace(target + ".attention.o_proj.weight",
                        WeightSource{source + ".self_attn.o_proj.weight",
                                     WeightTransform::Transpose2D});
        mapping.emplace(target + ".ffn_norm.weight",
                        WeightSource{source + ".post_attention_layernorm.weight",
                                     WeightTransform::Identity});
        mapping.emplace(target + ".feed_forward.gate_proj.weight",
                        WeightSource{source + ".mlp.gate_proj.weight",
                                     WeightTransform::Transpose2D});
        mapping.emplace(target + ".feed_forward.up_proj.weight",
                        WeightSource{source + ".mlp.up_proj.weight",
                                     WeightTransform::Transpose2D});
        mapping.emplace(target + ".feed_forward.down_proj.weight",
                        WeightSource{source + ".mlp.down_proj.weight",
                                     WeightTransform::Transpose2D});
    }
    mapping.emplace("final_norm.weight",
                    WeightSource{"model.norm.weight", WeightTransform::Identity});
    if (!config.tie_embeddings) {
        mapping.emplace("output_head.weight",
                        WeightSource{"lm_head.weight", WeightTransform::Transpose2D});
    }
    return mapping;
}

}  // namespace microllm::model
