#include <gtest/gtest.h>
#include <microllm/ops/ops.h>
#include <microllm/runtime/runtime.h>

namespace microllm::runtime {

TEST(HipPrecisionCapabilityTest, ReportsArchitectureSpecificNativeFormats) {
    if (hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto capabilities = precision_capabilities(Device::hip(0));
    RecordProperty("architecture", capabilities.architecture);
    RecordProperty("hip_runtime_version", hip_runtime_version());
    RecordProperty("hipblaslt_compiled", ops::hipblaslt_available());
    RecordProperty("fp8_fnuz", capabilities.fp8_fnuz);
    RecordProperty("fp8_ocp", capabilities.fp8_ocp);
    RecordProperty("mxfp4", capabilities.mxfp4);
    RecordProperty("int4_matrix", capabilities.int4_matrix);

    EXPECT_TRUE(capabilities.fp32);
    EXPECT_TRUE(capabilities.fp16);
    EXPECT_TRUE(capabilities.bfloat16);
    EXPECT_TRUE(capabilities.int8_matrix);
    EXPECT_TRUE(capabilities.packed_int4_software);

    if (capabilities.architecture.rfind("gfx942", 0) == 0) {
        EXPECT_TRUE(capabilities.fp64);
        EXPECT_TRUE(capabilities.tf32_hardware);
        EXPECT_TRUE(capabilities.fp8_fnuz);
        EXPECT_FALSE(capabilities.fp8_ocp);
        EXPECT_FALSE(capabilities.mxfp8);
        EXPECT_FALSE(capabilities.mxfp6);
        EXPECT_FALSE(capabilities.mxfp4);
        EXPECT_FALSE(capabilities.int4_matrix);
    } else if (capabilities.architecture.rfind("gfx950", 0) == 0) {
        EXPECT_TRUE(capabilities.fp64);
        EXPECT_FALSE(capabilities.tf32_hardware);
        EXPECT_FALSE(capabilities.fp8_fnuz);
        EXPECT_TRUE(capabilities.fp8_ocp);
        EXPECT_TRUE(capabilities.mxfp8);
        EXPECT_TRUE(capabilities.mxfp6);
        EXPECT_TRUE(capabilities.mxfp4);
        EXPECT_FALSE(capabilities.int4_matrix);
    } else {
        GTEST_SKIP() << "Capability table has no audited entry for "
                     << capabilities.architecture;
    }
}

TEST(HipPrecisionCapabilityTest, LibraryAndExecutedKernelEvidenceAreSeparate) {
    if (hip_device_count() == 0) GTEST_SKIP() << "No visible HIP device";
    const auto capabilities = precision_capabilities(Device::hip(0));
    // A header/library exposing a dtype is not proof of a native hardware matrix path.
    EXPECT_EQ(capabilities.mxfp4, capabilities.architecture.rfind("gfx950", 0) == 0);
    EXPECT_FALSE(capabilities.int4_matrix);
    EXPECT_EQ(ops::hipblaslt_available(), MICROLLM_HAS_HIPBLASLT != 0);
}

}  // namespace microllm::runtime
