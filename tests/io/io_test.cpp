#include <string>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include <microllm/io/byte_tokenizer.h>
#include <microllm/io/bpe_tokenizer.h>
#include <microllm/io/huggingface_bpe_tokenizer.h>
#include <microllm/io/chat_template.h>
#include <microllm/io/token_dataset.h>
#include <microllm/io/sft.h>

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

TEST(HuggingFaceBpeTokenizerTest, LoadsByteUnicodeVocabularyAndRankedMerges) {
    const auto directory = std::filesystem::temp_directory_path();
    const auto vocabulary = directory / "microllm-hf-vocab.json";
    const auto merges = directory / "microllm-hf-merges.txt";
    std::ofstream(vocabulary) <<
        "{\"H\":0,\"i\":1,\"Hi\":2,\"Ġ\":3,\"ĠĠ\":4,"
        "\"ĠH\":5,\"ĠHi\":6}";
    std::ofstream(merges) << "H i\nĠ Ġ\nĠ Hi\n";
    auto tokenizer = HuggingFaceBpeTokenizer::load(vocabulary, merges);
    tokenizer.add_special_token("<|im_start|>", 151644);
    EXPECT_EQ(tokenizer.encode("Hi"), (std::vector<std::int32_t>{2}));
    EXPECT_EQ(tokenizer.encode(" "), (std::vector<std::int32_t>{3}));
    EXPECT_EQ(tokenizer.encode("H  Hi"),
              (std::vector<std::int32_t>{0, 3, 6}));
    EXPECT_EQ(tokenizer.decode({2, 3}), "Hi ");
    EXPECT_EQ(tokenizer.encode("Hi<|im_start|>Hi"),
              (std::vector<std::int32_t>{2, 151644, 2}));
    EXPECT_EQ(tokenizer.decode({2, 151644, 2}), "Hi<|im_start|>Hi");
    EXPECT_THROW(tokenizer.add_special_token("<|im_start|>", 9), std::invalid_argument);
    EXPECT_THROW((void)tokenizer.decode({7}), std::out_of_range);
    std::error_code ignored;
    std::filesystem::remove(vocabulary, ignored);
    std::filesystem::remove(merges, ignored);
}

TEST(ChatTemplateTest, RendersBasicQwenConversationAndRejectsToolRole) {
    const auto prompt = render_qwen2_chat({{"user", "Hello"}});
    EXPECT_EQ(prompt,
              "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
              "helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n"
              "<|im_start|>assistant\n");
    EXPECT_EQ(render_qwen2_chat({{"system", "Be concise"}, {"user", "Hi"}}, false),
              "<|im_start|>system\nBe concise<|im_end|>\n"
              "<|im_start|>user\nHi<|im_end|>\n");
    EXPECT_THROW((void)render_qwen2_chat({{"tool", "result"}}), std::invalid_argument);
}

TEST(ChatTemplateTest, RendersDeepSeekDistillReasoningPrompt) {
    EXPECT_EQ(render_deepseek_distill_chat({{"user", "What is 2+2?"}}),
              "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?"
              "<｜Assistant｜><think>\n");
    EXPECT_EQ(render_deepseek_distill_chat(
                  {{"system", "Be precise"}, {"user", "Hi"},
                   {"assistant", "<think>hidden</think>Answer"}}, false),
              "<｜begin▁of▁sentence｜>Be precise<｜User｜>Hi"
              "<｜Assistant｜>Answer<｜end▁of▁sentence｜>");
}

TEST(SftBatchTest, MasksPromptTargetsAndKeepsResponseTargets) {
    const auto batch = make_sft_batch({1, 2, 3}, {4, 5});
    EXPECT_EQ(batch.inputs.to_int32_vector(), (std::vector<std::int32_t>{1, 2, 3, 4}));
    EXPECT_EQ(batch.targets.to_int32_vector(),
              (std::vector<std::int32_t>{-100, -100, 4, 5}));
    EXPECT_THROW((void)make_sft_batch({}, {1}), std::invalid_argument);
    EXPECT_THROW((void)make_sft_batch({1}, {}), std::invalid_argument);
}

}  // namespace microllm::io
