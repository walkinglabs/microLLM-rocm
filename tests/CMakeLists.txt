add_executable(microllm_tests
    tests_main.cpp
    core/device_test.cpp
    core/storage_test.cpp
    core/tensor_test.cpp
    core/headers_compile_test.cpp
    runtime/runtime_test.cpp
    profiling/trace_test.cpp
    ops/ops_test.cpp
    ops/property_test.cpp
    autograd/autograd_test.cpp
    graph/graph_construction_test.cpp
    graph/graph_gradient_alignment_test.cpp
    training/optimizer_test.cpp
    training/checkpoint_test.cpp
    model/config_test.cpp
    model/model_test.cpp
    model/weights_test.cpp
    io/io_test.cpp
    io/safetensors_test.cpp
    training/trainer_test.cpp
    inference/generator_test.cpp
    inference/scheduler_test.cpp
)

target_link_libraries(microllm_tests PRIVATE
    GTest::gtest_main microllm::core microllm::ops microllm::autograd
    microllm::training microllm::model microllm::io
    microllm::inference
)
target_compile_definitions(microllm_tests PRIVATE
    MICROLLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

add_executable(microllm_safetensors_interop
    io/safetensors_interop_helper.cpp
)
target_link_libraries(microllm_safetensors_interop PRIVATE microllm::io)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME Coverage.OperatorAndTestFiles
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/audit_test_coverage.py
    )
    set_tests_properties(Coverage.OperatorAndTestFiles PROPERTIES LABELS "cpu;coverage;schema")
    add_test(
        NAME Alignment.Tools
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/python/tests/test_alignment_tools.py
    )
    set_tests_properties(Alignment.Tools PROPERTIES LABELS "cpu;alignment;schema")
endif()

if(MICROLLM_SAFETENSORS_PYTHON)
    if(NOT EXISTS "${MICROLLM_SAFETENSORS_PYTHON}")
        message(FATAL_ERROR
            "MICROLLM_SAFETENSORS_PYTHON does not exist: ${MICROLLM_SAFETENSORS_PYTHON}")
    endif()
    add_test(
        NAME Safetensors.OfficialInterop
        COMMAND ${MICROLLM_SAFETENSORS_PYTHON}
            ${PROJECT_SOURCE_DIR}/tests/io/official_safetensors_interop.py
            --helper $<TARGET_FILE:microllm_safetensors_interop>
    )
    set_tests_properties(Safetensors.OfficialInterop PROPERTIES
        LABELS "cpu;weights;interop;external"
    )
endif()

include(GoogleTest)
gtest_discover_tests(microllm_tests
    PROPERTIES LABELS "cpu\;unit"
    DISCOVERY_TIMEOUT 30
)

if(MICROLLM_HAS_HIP)
    add_executable(microllm_hip_tests
        hardware/precision_capability_test.cpp
        runtime/runtime_test.cpp
        ops/hip_ops_test.cpp
        graph/hip_graph_alignment_test.cpp
        io/hip_weights_test.cpp
        profiling/hip_trace_test.cpp
    )
    target_link_libraries(microllm_hip_tests PRIVATE
        GTest::gtest_main microllm::ops microllm::model microllm::training
        microllm::inference
    )
    gtest_discover_tests(microllm_hip_tests
        TEST_PREFIX "HIP."
        PROPERTIES LABELS "hip\;conformance"
        DISCOVERY_TIMEOUT 30
    )
endif()

if(MICROLLM_HAS_RCCL)
    add_executable(microllm_multi_gpu_tests
        multi_gpu/communicator_test.cpp
        multi_gpu/distributed_step_test.cpp
        multi_gpu/gradient_bucket_test.cpp
        multi_gpu/data_parallel_test.cpp
    )
    target_link_libraries(microllm_multi_gpu_tests PRIVATE
        GTest::gtest_main microllm::multi_gpu microllm::training microllm::model
    )
    gtest_discover_tests(microllm_multi_gpu_tests
        TEST_PREFIX "RCCL."
        PROPERTIES LABELS "hip\;rccl\;multi_gpu"
        DISCOVERY_TIMEOUT 60
    )
endif()
