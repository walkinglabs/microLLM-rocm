#include <vector>

#include <gtest/gtest.h>
#include <microllm/multi_gpu/communicator.h>
#include <microllm/runtime/runtime.h>

namespace microllm::multi_gpu {

TEST(RcclCommunicatorTest, TwoGpuAverageProducesIdenticalValues) {
    if (runtime::hip_device_count() < 2) GTEST_SKIP() << "two visible HIP devices required";
    Communicator communicator({0, 1});
    std::vector<Tensor> tensors;
    tensors.push_back(Tensor::from_vector({1, 2, 3, 4}, {4}).to(Device::hip(0)));
    tensors.push_back(Tensor::from_vector({3, 4, 5, 6}, {4}).to(Device::hip(1)));
    const auto* first_address = tensors[0].storage().data();
    const auto* second_address = tensors[1].storage().data();
    communicator.all_reduce(tensors, true);
    EXPECT_EQ(tensors[0].storage().data(), first_address);
    EXPECT_EQ(tensors[1].storage().data(), second_address);
    EXPECT_EQ(tensors[0].to_vector(), (std::vector<float>{2, 3, 4, 5}));
    EXPECT_EQ(tensors[1].to_vector(), tensors[0].to_vector());
    EXPECT_FALSE(communicator.aborted());
}

TEST(RcclCommunicatorTest, AllocatingAverageControlMatchesInPlaceValues) {
    if (runtime::hip_device_count() < 2) GTEST_SKIP() << "two visible HIP devices required";
    Communicator communicator({0, 1});
    std::vector<Tensor> tensors;
    tensors.push_back(Tensor::from_vector({1, 2, 3, 4}, {4}).to(Device::hip(0)));
    tensors.push_back(Tensor::from_vector({3, 4, 5, 6}, {4}).to(Device::hip(1)));
    communicator.all_reduce(tensors, true, false);
    EXPECT_EQ(tensors[0].to_vector(), (std::vector<float>{2, 3, 4, 5}));
    EXPECT_EQ(tensors[1].to_vector(), tensors[0].to_vector());
}

TEST(RcclCommunicatorTest, ValidatesAllRanksBeforeCommunication) {
    if (runtime::hip_device_count() < 2) GTEST_SKIP() << "two visible HIP devices required";
    Communicator communicator({0, 1});
    std::vector<Tensor> tensors;
    tensors.push_back(Tensor({2}, DType::Float32, Device::hip(0)));
    tensors.push_back(Tensor({3}, DType::Float32, Device::hip(1)));
    EXPECT_THROW(communicator.all_reduce(tensors), std::invalid_argument);
    EXPECT_FALSE(communicator.aborted());
}

TEST(RcclCommunicatorTest, AsyncSumCompletesWhenCommunicationStreamsSynchronize) {
    if (runtime::hip_device_count() < 2) GTEST_SKIP() << "two visible HIP devices required";
    Communicator communicator({0, 1});
    std::vector<Tensor> tensors;
    tensors.push_back(Tensor::from_vector({1, 2}, {2}).to(Device::hip(0)));
    tensors.push_back(Tensor::from_vector({3, 4}, {2}).to(Device::hip(1)));
    communicator.enqueue_all_reduce_sum(tensors);
    communicator.synchronize();
    EXPECT_EQ(tensors[0].to_vector(), (std::vector<float>{4, 6}));
    EXPECT_EQ(tensors[1].to_vector(), tensors[0].to_vector());
}

TEST(RcclCommunicatorTest, AsyncAverageIsOrderedOnCommunicationStreams) {
    if (runtime::hip_device_count() < 2) GTEST_SKIP() << "two visible HIP devices required";
    Communicator communicator({0, 1});
    std::vector<Tensor> tensors;
    tensors.push_back(Tensor::from_vector({1, 2}, {2}).to(Device::hip(0)));
    tensors.push_back(Tensor::from_vector({3, 4}, {2}).to(Device::hip(1)));
    communicator.enqueue_all_reduce_average_in_place(tensors);
    communicator.synchronize();
    EXPECT_EQ(tensors[0].to_vector(), (std::vector<float>{2, 3}));
    EXPECT_EQ(tensors[1].to_vector(), tensors[0].to_vector());
}

TEST(RcclRankCommunicatorTest, WorldOneRoundTripAndIdentityValidation) {
    if (runtime::hip_device_count() < 1) GTEST_SKIP() << "visible HIP device required";
    const auto id = create_communicator_id();
    EXPECT_EQ(id.size(), communicator_id_bytes());
    EXPECT_GT(id.size(), 0U);
    RankCommunicator communicator(0, 1, 0, id);
    auto tensor = Tensor::from_vector({1, 2, 3}, {3}).to(Device::hip(0));
    communicator.enqueue_all_reduce_average_in_place(tensor);
    communicator.synchronize();
    EXPECT_EQ(tensor.to_vector(), (std::vector<float>{1, 2, 3}));
    EXPECT_EQ(communicator.rank(), 0);
    EXPECT_EQ(communicator.world_size(), 1);
    EXPECT_EQ(communicator.device(), Device::hip(0));
    EXPECT_FALSE(communicator.aborted());

    EXPECT_THROW((void)RankCommunicator(-1, 1, 0, id),
                 std::invalid_argument);
    EXPECT_THROW((void)RankCommunicator(1, 1, 0, id),
                 std::invalid_argument);
    auto short_id = id;
    short_id.pop_back();
    EXPECT_THROW((void)RankCommunicator(0, 1, 0, short_id),
                 std::invalid_argument);
}

}  // namespace microllm::multi_gpu
