#include <array>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include <microllm/core/storage.h>
#include <microllm/core/tensor.h>

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

TEST(StorageTest, ExternalStorageIsNonOwningAndSharesCallerBytes) {
    std::array<float, 4> values{1, 2, 3, 4};
    {
        auto external = Storage::from_external(
            values.data(), sizeof(values), Device::cpu());
        EXPECT_EQ(external.data(), values.data());
        EXPECT_EQ(external.num_bytes(), sizeof(values));
        auto tensor = Tensor::from_storage(
            external, {2, 2}, {2, 1}, 0, DType::Float32);
        tensor.fill(7.0F);
        EXPECT_EQ(values, (std::array<float, 4>{7, 7, 7, 7}));
    }
    values[0] = 9.0F;
    EXPECT_EQ(values[0], 9.0F);
    EXPECT_THROW(
        (void)Storage::from_external(nullptr, 4, Device::cpu()),
        std::invalid_argument);
}

TEST(StorageTest, HipAllocationIsExplicitlyUnavailableInN0) {
#if MICROLLM_HAS_HIP
    GTEST_SKIP() << "HIP allocation is covered by runtime tests in HIP builds";
#else
    EXPECT_THROW((void)Storage(4, Device::hip()), std::runtime_error);
#endif
}

}  // namespace microllm
