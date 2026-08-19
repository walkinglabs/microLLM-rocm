#include <microllm/model/model.h>

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace microllm::model {
namespace {

using autograd::Value;

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

Value TransformerModel::forward(const Tensor& token_ids) {
    if (!token_ids.device().is_cpu() || token_ids.dtype() != DType::Int32 ||
        token_ids.ndim() != 2) {
        throw std::invalid_argument("model token IDs must be a CPU int32 BxT tensor");
    }
    if (token_ids.shape()[1] > impl_->config.max_sequence_length) {
        throw std::invalid_argument("token sequence exceeds configured maximum");
    }
    auto hidden = autograd::embedding(impl_->token_embedding, token_ids);
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
    return autograd::cross_entropy(forward(token_ids), targets);
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
