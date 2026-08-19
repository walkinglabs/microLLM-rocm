#include <microllm/io/token_dataset.h>

#include <sstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace microllm::io {

TokenDataset::TokenDataset(std::vector<std::int32_t> tokens, std::int64_t context_length)
    : tokens_(std::move(tokens)), context_length_(context_length) {
    if (context_length_ <= 0) throw std::invalid_argument("context length must be positive");
    if (tokens_.size() <= static_cast<std::size_t>(context_length_)) {
        throw std::invalid_argument("dataset needs at least context_length + 1 tokens");
    }
    for (const auto token : tokens_) {
        if (token < 0) throw std::invalid_argument("dataset tokens must be non-negative");
    }
}

std::uint64_t TokenDataset::valid_start_count() const noexcept {
    return tokens_.size() - static_cast<std::size_t>(context_length_);
}

TokenBatch TokenDataset::next_batch(std::int64_t batch_size) {
    if (batch_size <= 0) throw std::invalid_argument("batch size must be positive");
    if (batch_size > std::numeric_limits<std::int64_t>::max() / context_length_) {
        throw std::overflow_error("batch element count overflows int64");
    }
    std::vector<std::int32_t> inputs(
        static_cast<std::size_t>(batch_size * context_length_));
    std::vector<std::int32_t> targets(inputs.size());
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
        const auto start = cursor_;
        for (std::int64_t position = 0; position < context_length_; ++position) {
            const auto destination = static_cast<std::size_t>(batch * context_length_ + position);
            const auto source = static_cast<std::size_t>(start) + static_cast<std::size_t>(position);
            inputs[destination] = tokens_[source];
            targets[destination] = tokens_[source + 1];
        }
        cursor_ = (cursor_ + static_cast<std::uint64_t>(context_length_)) % valid_start_count();
    }
    return {Tensor::from_int32_vector(inputs, {batch_size, context_length_}),
            Tensor::from_int32_vector(targets, {batch_size, context_length_})};
}

void TokenDataset::set_cursor(std::uint64_t cursor) {
    if (cursor >= valid_start_count()) throw std::out_of_range("dataset cursor is out of range");
    cursor_ = cursor;
}

std::string TokenDataset::summary() const {
    std::ostringstream output;
    output << "tokens=" << tokens_.size() << ",context=" << context_length_
           << ",valid_starts=" << valid_start_count();
    return output.str();
}

}  // namespace microllm::io
