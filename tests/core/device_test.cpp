#include <stdexcept>

#include <gtest/gtest.h>
#include <microllm/base/device.h>

namespace microllm {

TEST(DeviceTest, DescribesCpuAndHipDevices) {
    EXPECT_EQ(Device::cpu().str(), "cpu:0");
    EXPECT_EQ(Device::hip(3).str(), "hip:3");
    EXPECT_TRUE(Device::cpu().is_cpu());
    EXPECT_TRUE(Device::hip().is_hip());
    EXPECT_NE(Device::cpu(), Device::hip());
}

TEST(DeviceTest, RejectsNegativeIndex) {
    EXPECT_THROW((void)Device::cpu(-1), std::invalid_argument);
    EXPECT_THROW((void)Device::hip(-1), std::invalid_argument);
}

}  // namespace microllm
