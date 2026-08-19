#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include <microllm/core/storage.h>

namespace microllm {

TEST(StorageTest, AllocatesCpuBytes) {
    Storage storage(64);
    EXPECT_NE(storage.data(), nullptr);
    EXPECT_EQ(storage.num_bytes(), 64U);
    EXPECT_EQ(storage.device(), Device::cpu());
    EXPECT_EQ(storage.use_count(), 1);
}

TEST(StorageTest, CopySharesTheAllocationLifetime) {
    Storage owner(sizeof(std::int32_t));
    *static_cast<std::int32_t*>(owner.data()) = 42;
    Storage alias = owner;
    EXPECT_EQ(owner.use_count(), 2);
    EXPECT_EQ(alias.data(), owner.data());
    owner = Storage();
    EXPECT_EQ(*static_cast<const std::int32_t*>(alias.data()), 42);
    EXPECT_EQ(alias.use_count(), 1);
}

TEST(StorageTest, ZeroByteStorageIsValidAndEmpty) {
    Storage storage(0);
    EXPECT_TRUE(storage.empty());
    EXPECT_EQ(storage.num_bytes(), 0U);
}

TEST(StorageTest, HipAllocationIsExplicitlyUnavailableInN0) {
#if MICROLLM_HAS_HIP
    GTEST_SKIP() << "HIP allocation is covered by runtime tests in HIP builds";
#else
    EXPECT_THROW((void)Storage(4, Device::hip()), std::runtime_error);
#endif
}

}  // namespace microllm
