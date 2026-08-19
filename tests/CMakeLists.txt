add_executable(microllm_tests
    tests_main.cpp
    core/device_test.cpp
    core/storage_test.cpp
    core/tensor_test.cpp
    core/headers_compile_test.cpp
    runtime/runtime_test.cpp
)

target_link_libraries(microllm_tests PRIVATE GTest::gtest_main microllm::core)

include(GoogleTest)
gtest_discover_tests(microllm_tests
    PROPERTIES LABELS "cpu;unit"
    DISCOVERY_TIMEOUT 30
)
