#include <microllm/model/model.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include <microllm/ops/ops.h>

namespace microllm::model {
namespace {

using autograd::Value;

Tensor append_cached_sequence(const Tensor& cached, const Tensor& current) {
    if (current.dtype() != DType::Float32 || current.ndim() != 4) {
        throw std::invalid_argument("cached K/V tensors must be float32 rank four");
    }
    if (!cached.defined()) return current.contiguous();
    if (cached.ndim() != 4 || cached.shape()[0] != current.shape()[0] ||
        cached.shape()[1] != current.shape()[1] || cached.shape()[3] != current.shape()[3]) {
        throw std::invalid_argument("cached and current K/V shapes are incompatible");
    }
    const auto batch = current.shape()[0];
    const auto heads = current.shape()[1];
    const auto old_sequence = cached.shape()[2];
    const auto new_sequence = current.shape()[2];
    const auto width = current.shape()[3];
    const auto old_values = cached.to_vector();
    const auto new_values = current.to_vector();
    std::vector<float> output(static_cast<std::size_t>(
        batch * heads * (old_sequence + new_sequence) * width));
    for (std::int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (std::int64_t head = 0; head < heads; ++head) {
            const auto old_base = (batch_index * heads + head) * old_sequence * width;
            const auto new_base = (batch_index * heads + head) * new_sequence * width;
            const auto output_base =
                (batch_index * heads + head) * (old_sequence + new_sequence) * width;
            std::copy_n(old_values.begin() + old_base, old_sequence * width,
                        output.begin() + output_base);
            std::copy_n(new_values.begin() + new_base, new_sequence * width,
                        output.begin() + output_base + old_sequence * width);
        }
    }
    return Tensor::from_vector(output, {batch, heads, old_sequence + new_sequence, width})
        .to(current.device());
}

Tensor random_tensor(Shape shape, std::mt19937_64& generator, float standard_deviation) {
    std::normal_distribution<float> distribution(0.0F, standard_deviation);
    std::vector<float> values(static_cast<std::size_t>(checked_numel(shape)));
    for (auto& value : values) value = distribution(generator);
    return Tensor::from_vector(values, std::move(shape));
}

Value parameter(Shape shape, std::mt19937_64& generator, float standard_deviation) {
    return Value(random_tensor(std::move(shape), generator, standard_deviation), true);
}

class Linear {
public:
    Linear(std::int64_t input, std::int64_t output, std::mt19937_64& generator)
        : weight_(parameter({input, output}, generator,
                            1.0F / std::sqrt(static_cast<float>(input)))) {}

    Value forward(const Value& input) { return autograd::matmul(input, weight_); }
    Tensor forward_tensor(const Tensor& input) { return ops::matmul(input, weight_.data()); }
    Value& weight() noexcept { return weight_; }

private:
    Value weight_;
};

class Norm {
public:
    explicit Norm(std::int64_t dimension) : weight_(Tensor({dimension}), true) {
        weight_.mutable_data().fill(1.0F);
    }

    Value forward(const Value& input) { return autograd::rms_norm(input, weight_); }
    Tensor forward_tensor(const Tensor& input) { return ops::rms_norm(input, weight_.data()); }
    Value& weight() noexcept { return weight_; }

private:
    Value weight_;
};

class Attention {
public:
    Attention(const ModelConfig& config, std::mt19937_64& generator)
        : config_(config),
          query_(config.dimension, config.dimension, generator),
          key_(config.dimension, config.kv_dimension(), generator),
          value_(config.dimension, config.kv_dimension(), generator),
          output_(config.dimension, config.dimension, generator) {}

    Value forward(const Value& input) {
        if (input.data().ndim() != 3) throw std::invalid_argument("attention input must be BxTxD");
        const auto batch = input.data().shape()[0];
        const auto sequence = input.data().shape()[1];
        const auto flat = autograd::reshape(input, {batch * sequence, config_.dimension});
        auto query = autograd::reshape(query_.forward(flat),
                                       {batch, sequence, config_.heads, config_.head_dimension()});
        auto key = autograd::reshape(key_.forward(flat),
                                     {batch, sequence, config_.kv_heads, config_.head_dimension()});
        auto value = autograd::reshape(value_.forward(flat),
                                       {batch, sequence, config_.kv_heads, config_.head_dimension()});
        query = autograd::rope(autograd::transpose(query, 1, 2), 2, 0, config_.rope_base);
        key = autograd::rope(autograd::transpose(key, 1, 2), 2, 0, config_.rope_base);
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

    Tensor forward_cached(const Tensor& input, inference::KVCache::LayerState& cache,
                          std::int64_t position) {
        if (input.shape().size() != 3 || input.shape()[0] != 1 || input.shape()[1] != 1) {
            throw std::invalid_argument("cached attention expects one B=1 token");
        }
        auto query = query_.forward_tensor(input.reshape({1, config_.dimension}))
                         .reshape({1, 1, config_.heads, config_.head_dimension()})
                         .transpose(1, 2);
        auto key = key_.forward_tensor(input.reshape({1, config_.dimension}))
                       .reshape({1, 1, config_.kv_heads, config_.head_dimension()})
                       .transpose(1, 2);
        auto value = value_.forward_tensor(input.reshape({1, config_.dimension}))
                         .reshape({1, 1, config_.kv_heads, config_.head_dimension()})
                         .transpose(1, 2);
        query = ops::rope(query, 2, position, config_.rope_base);
        key = ops::rope(key, 2, position, config_.rope_base);
        cache.key = append_cached_sequence(cache.key, key);
        cache.value = append_cached_sequence(cache.value, value);
        auto expanded_key = cache.key;
        auto expanded_value = cache.value;
        if (config_.kv_heads != config_.heads) {
            const auto repeats = config_.heads / config_.kv_heads;
            const auto old_key = cache.key.to_vector();
            const auto old_value = cache.value.to_vector();
            const auto cached_sequence = cache.key.shape()[2];
            std::vector<float> key_values(static_cast<std::size_t>(
                config_.heads * cached_sequence * config_.head_dimension()));
            std::vector<float> value_values(key_values.size());
            for (std::int64_t head = 0; head < config_.heads; ++head) {
                const auto source_head = head / repeats;
                const auto source_base = source_head * cached_sequence * config_.head_dimension();
                const auto destination_base = head * cached_sequence * config_.head_dimension();
                std::copy_n(old_key.begin() + source_base,
                            cached_sequence * config_.head_dimension(),
                            key_values.begin() + destination_base);
                std::copy_n(old_value.begin() + source_base,
                            cached_sequence * config_.head_dimension(),
                            value_values.begin() + destination_base);
            }
            expanded_key = Tensor::from_vector(
                               key_values,
                               {1, config_.heads, cached_sequence, config_.head_dimension()})
                               .to(cache.key.device());
            expanded_value = Tensor::from_vector(
                                 value_values,
                                 {1, config_.heads, cached_sequence, config_.head_dimension()})
                                 .to(cache.value.device());
        }
        const auto scores = ops::scale(
            ops::matmul(query, expanded_key.transpose(-2, -1)),
            1.0F / std::sqrt(static_cast<float>(config_.head_dimension())));
        const auto probabilities = ops::softmax(scores);
        auto context = ops::matmul(probabilities, expanded_value)
                           .transpose(1, 2)
                           .contiguous()
                           .reshape({1, config_.dimension});
        return output_.forward_tensor(context).reshape({1, 1, config_.dimension});
    }

    void append_named(const std::string& prefix, NamedValues& values) {
        values.emplace_back(prefix + ".q_proj.weight", &query_.weight());
        values.emplace_back(prefix + ".k_proj.weight", &key_.weight());
        values.emplace_back(prefix + ".v_proj.weight", &value_.weight());
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
          gate_(config.dimension, config.ffn_dimension, generator),
          up_(config.dimension, config.ffn_dimension, generator),
          down_(config.ffn_dimension, config.dimension, generator) {}

    Value forward(const Value& input) {
        const auto batch = input.data().shape()[0];
        const auto sequence = input.data().shape()[1];
        const auto flat = autograd::reshape(input, {batch * sequence, config_.dimension});
        const auto activated = autograd::swiglu(gate_.forward(flat), up_.forward(flat));
        return autograd::reshape(down_.forward(activated),
                                 {batch, sequence, config_.dimension});
    }

    Tensor forward_tensor(const Tensor& input) {
        const auto flat = input.reshape({1, config_.dimension});
        const auto activated = ops::swiglu(gate_.forward_tensor(flat), up_.forward_tensor(flat));
        return down_.forward_tensor(activated).reshape({1, 1, config_.dimension});
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
        : attention_norm_(config.dimension),
          attention_(config, generator),
          ffn_norm_(config.dimension),
          feed_forward_(config, generator) {}

    Value forward(const Value& input) {
        auto hidden = autograd::add(input, attention_.forward(attention_norm_.forward(input)));
        return autograd::add(hidden, feed_forward_.forward(ffn_norm_.forward(hidden)));
    }

    Tensor forward_cached(const Tensor& input, inference::KVCache::LayerState& cache,
                          std::int64_t position) {
        auto hidden = ops::add(
            input, attention_.forward_cached(attention_norm_.forward_tensor(input), cache, position));
        return ops::add(hidden, feed_forward_.forward_tensor(ffn_norm_.forward_tensor(hidden)));
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
          final_norm(config.dimension) {
        config.validate();
        blocks.reserve(static_cast<std::size_t>(config.layers));
        for (std::int64_t layer = 0; layer < config.layers; ++layer) {
            blocks.push_back(std::make_unique<Block>(config, generator));
        }
        if (!config.tie_embeddings) {
            output_head = std::make_unique<Linear>(config.dimension, config.vocabulary_size,
                                                   generator);
        }
    }

    ModelConfig config;
    std::mt19937_64 generator;
    Value token_embedding;
    std::vector<std::unique_ptr<Block>> blocks;
    Norm final_norm;
    std::unique_ptr<Linear> output_head;
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
    if (token_ids.dtype() != DType::Int32 || token_ids.ndim() != 2) {
        throw std::invalid_argument("model token IDs must be an int32 BxT tensor");
    }
    if (token_ids.shape()[1] > impl_->config.max_sequence_length) {
        throw std::invalid_argument("token sequence exceeds configured maximum");
    }
    const auto model_tokens = token_ids.device() == device() ? token_ids : token_ids.to(device());
    auto hidden = autograd::embedding(impl_->token_embedding, model_tokens);
    for (auto& block : impl_->blocks) hidden = block->forward(hidden);
    hidden = impl_->final_norm.forward(hidden);
    const auto batch = token_ids.shape()[0];
    const auto sequence = token_ids.shape()[1];
    const auto flat = autograd::reshape(hidden, {batch * sequence, impl_->config.dimension});
    Value logits;
    if (impl_->config.tie_embeddings) {
        logits = autograd::matmul(flat, autograd::transpose(impl_->token_embedding, 0, 1));
    } else {
        logits = impl_->output_head->forward(flat);
    }
    return autograd::reshape(logits, {batch, sequence, impl_->config.vocabulary_size});
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
        cache.max_sequence_length() != impl_->config.max_sequence_length) {
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
                                                      cache.position());
    }
    hidden = impl_->final_norm.forward_tensor(hidden);
    const auto flat = hidden.reshape({1, impl_->config.dimension});
    Tensor logits;
    if (impl_->config.tie_embeddings) {
        logits = ops::matmul(flat, impl_->token_embedding.data().transpose(0, 1));
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

}  // namespace microllm::model
