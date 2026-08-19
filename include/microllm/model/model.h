#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <microllm/autograd/autograd.h>
#include <microllm/model/config.h>
#include <microllm/inference/kv_cache.h>

namespace microllm::model {

using NamedValues = std::vector<std::pair<std::string, autograd::Value*>>;

class TransformerModel {
public:
    explicit TransformerModel(ModelConfig config, std::uint64_t seed = 1);
    ~TransformerModel();
    TransformerModel(TransformerModel&&) noexcept;
    TransformerModel& operator=(TransformerModel&&) noexcept;
    TransformerModel(const TransformerModel&) = delete;
    TransformerModel& operator=(const TransformerModel&) = delete;

    [[nodiscard]] const ModelConfig& config() const noexcept;
    [[nodiscard]] autograd::Value forward(const Tensor& token_ids);
    [[nodiscard]] Tensor forward_cached(const Tensor& token_id,
                                        inference::KVCache& cache);
    [[nodiscard]] autograd::Value loss(const Tensor& token_ids, const Tensor& targets);
    [[nodiscard]] NamedValues named_parameters();
    [[nodiscard]] std::vector<autograd::Value*> parameters();
    [[nodiscard]] std::uint64_t parameter_count();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace microllm::model
