#include <microllm/io/sft.h>

#include <stdexcept>

namespace microllm::io {

TokenBatch make_sft_batch(const std::vector<std::int32_t>& prompt,
                          const std::vector<std::int32_t>& response) {
    if (prompt.empty()) throw std::invalid_argument("SFT prompt cannot be empty");
    if (response.empty()) throw std::invalid_argument("SFT response cannot be empty");
    std::vector<std::int32_t> sequence = prompt;
    sequence.insert(sequence.end(), response.begin(), response.end());
    std::vector<std::int32_t> inputs(sequence.begin(), sequence.end() - 1);
    std::vector<std::int32_t> targets(sequence.begin() + 1, sequence.end());
    for (std::size_t position = 0; position < targets.size(); ++position) {
        const auto predicted_sequence_position = position + 1;
        if (predicted_sequence_position < prompt.size()) targets[position] = kIgnoredTarget;
    }
    return {Tensor::from_int32_vector(inputs, {1, static_cast<std::int64_t>(inputs.size())}),
            Tensor::from_int32_vector(targets, {1, static_cast<std::int64_t>(targets.size())})};
}

TokenBatch make_sft_text_batch(std::string_view prompt, std::string_view response,
                               const ByteTokenizer& tokenizer) {
    const auto formatted_prompt = std::string("User: ") + std::string(prompt) + "\nAssistant: ";
    return make_sft_batch(tokenizer.encode(formatted_prompt), tokenizer.encode(response));
}

}  // namespace microllm::io
