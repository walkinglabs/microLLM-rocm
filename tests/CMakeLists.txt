add_executable(microllm_tests
    tests_main.cpp
    core/device_test.cpp
    core/storage_test.cpp
    core/tensor_test.cpp
    core/headers_compile_test.cpp
    runtime/runtime_test.cpp
    ops/ops_test.cpp
    autograd/autograd_test.cpp
    training/optimizer_test.cpp
    training/checkpoint_test.cpp
)

target_link_libraries(microllm_tests PRIVATE
    GTest::gtest_main microllm::core microllm::ops microllm::autograd microllm::training
)

include(GoogleTest)
gtest_discover_tests(microllm_tests
    PROPERTIES LABELS "cpu;unit"
    DISCOVERY_TIMEOUT 30
)

if(MICROLLM_HAS_HIP)
    add_executable(microllm_hip_tests
        ops/hip_ops_test.cpp
    )
    target_link_libraries(microllm_hip_tests PRIVATE GTest::gtest_main microllm::ops)
    gtest_discover_tests(microllm_hip_tests
        TEST_PREFIX "HIP."
        PROPERTIES LABELS "hip;conformance"
        DISCOVERY_TIMEOUT 30
    )
endif()
