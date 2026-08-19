#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/io/byte_tokenizer.h>
#include <microllm/io/bpe_tokenizer.h>
#include <microllm/io/token_dataset.h>

namespace microllm::io {

TEST(ByteTokenizerTest, RoundTripsAllByteValues) {
    std::string bytes;
    for (int value = 0; value < 256; ++value) bytes.push_back(static_cast<char>(value));
    const ByteTokenizer tokenizer;
    const auto tokens = tokenizer.encode(bytes);
    EXPECT_EQ(tokens.size(), 256U);
    EXPECT_EQ(tokenizer.decode(tokens), bytes);
    EXPECT_THROW((void)tokenizer.decode({256}), std::out_of_range);
}

TEST(TokenDatasetTest, ProducesShiftedDeterministicBatches) {
    TokenDataset dataset({0, 1, 2, 3, 4, 5, 6, 7}, 3);
    const auto first = dataset.next_batch(2);
    EXPECT_EQ(first.inputs.to_int32_vector(), (std::vector<std::int32_t>{0, 1, 2, 3, 4, 5}));
    EXPECT_EQ(first.targets.to_int32_vector(), (std::vector<std::int32_t>{1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(dataset.cursor(), 1U);
}

TEST(TokenDatasetTest, RestoredCursorProducesTheSameNextBatch) {
    TokenDataset first({0, 1, 2, 3, 4, 5, 6, 7, 8}, 3);
    (void)first.next_batch(1);
    TokenDataset restored(first.tokens(), first.context_length());
    restored.set_cursor(first.cursor());
    EXPECT_EQ(restored.next_batch(2).inputs.to_int32_vector(),
              first.next_batch(2).inputs.to_int32_vector());
}

TEST(TokenDatasetTest, RejectsInsufficientDataAndBadCursor) {
    EXPECT_THROW((void)TokenDataset({1, 2, 3}, 3), std::invalid_argument);
    TokenDataset dataset({1, 2, 3, 4}, 2);
    EXPECT_THROW(dataset.set_cursor(2), std::out_of_range);
}

TEST(BpeTokenizerTest, LearnsFrequentPairsAndRoundTripsBytes) {
    const std::string text = "banana banana banana\n";
    const auto tokenizer = BpeTokenizer::train(text, 264);
    EXPECT_GT(tokenizer.vocabulary_size(), 256U);
    EXPECT_LT(tokenizer.encode(text).size(), text.size());
    EXPECT_EQ(tokenizer.decode(tokenizer.encode(text)), text);
    const auto restored = BpeTokenizer::deserialize(tokenizer.serialize());
    EXPECT_EQ(restored.encode(text), tokenizer.encode(text));
    EXPECT_EQ(restored.decode(restored.encode(text)), text);
}

TEST(BpeTokenizerTest, RejectsBadVocabularyAndSerializedMerge) {
    EXPECT_THROW((void)BpeTokenizer::train("abc", 255), std::invalid_argument);
    EXPECT_THROW((void)BpeTokenizer::deserialize("wrong\n"), std::invalid_argument);
    EXPECT_THROW((void)BpeTokenizer::deserialize("MICROLLM_BPE_V1\n999 1\n"),
                 std::invalid_argument);
}

}  // namespace microllm::io
