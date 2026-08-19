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
    communicator.all_reduce(tensors, true);
    EXPECT_EQ(tensors[0].to_vector(), (std::vector<float>{2, 3, 4, 5}));
    EXPECT_EQ(tensors[1].to_vector(), tensors[0].to_vector());
    EXPECT_FALSE(communicator.aborted());
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

}  // namespace microllm::multi_gpu
