#include <gtest/gtest.h>

TEST(BuildSmoke, VersionIsDefined) {
    EXPECT_STRNE(MICROLLM_VERSION, "");
    EXPECT_STRNE(MICROLLM_VERSION, "unknown");
}
