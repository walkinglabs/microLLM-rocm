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
    inference/shape_matrix_test.cpp
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

add_test(
    NAME PackageConfig.InstalledConsumer
    COMMAND ${CMAKE_COMMAND}
        -DMICROLLM_BINARY_DIR=${PROJECT_BINARY_DIR}
        -DMICROLLM_CONSUMER_SOURCE_DIR=${PROJECT_SOURCE_DIR}/tests/package/consumer
        -DMICROLLM_CONSUMER_BINARY_DIR=${PROJECT_BINARY_DIR}/package-consumer-build
        -DMICROLLM_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/package-consumer-prefix
        -P ${PROJECT_SOURCE_DIR}/cmake/RunInstalledConsumer.cmake
)
set(microllm_package_test_labels cpu package install integration)
if(MICROLLM_HAS_HIP)
    list(APPEND microllm_package_test_labels hip)
endif()
if(MICROLLM_HAS_RCCL)
    list(APPEND microllm_package_test_labels rccl)
endif()
set_tests_properties(PackageConfig.InstalledConsumer PROPERTIES
    LABELS "${microllm_package_test_labels}"
    RUN_SERIAL TRUE
)

add_test(
    NAME PackageConfig.BuildTreeConsumer
    COMMAND ${CMAKE_COMMAND}
        -DMICROLLM_BINARY_DIR=${PROJECT_BINARY_DIR}
        -DMICROLLM_CONSUMER_SOURCE_DIR=${PROJECT_SOURCE_DIR}/tests/package/consumer
        -DMICROLLM_CONSUMER_BINARY_DIR=${PROJECT_BINARY_DIR}/package-build-tree-consumer
        -P ${PROJECT_SOURCE_DIR}/cmake/RunBuildTreeConsumer.cmake
)
set_tests_properties(PackageConfig.BuildTreeConsumer PROPERTIES
    LABELS "${microllm_package_test_labels}"
    RUN_SERIAL TRUE
)

if(Python3_Interpreter_FOUND)
    add_test(
        NAME HfCLI.BinaryContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_hf_cli_binary_contract.py
                --binary $<TARGET_FILE:microllm_hf_infer>
    )
    set_tests_properties(HfCLI.BinaryContract PROPERTIES
        LABELS "cpu;cli;binary;schema")
    add_test(
        NAME Benchmark.HfFp8NativeRoundtripContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_hf_fp8_native_roundtrip.py
    )
    set_tests_properties(Benchmark.HfFp8NativeRoundtripContract PROPERTIES
        LABELS "cpu;benchmark;fp8;precision;schema")
    add_test(
        NAME Benchmark.HfFp8WeightAuditContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_hf_fp8_weight_audit.py
    )
    set_tests_properties(Benchmark.HfFp8WeightAuditContract PROPERTIES
        LABELS "cpu;benchmark;fp8;precision;schema")
    add_test(
        NAME Benchmark.AttentionLayoutFusionRunnerContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_attention_layout_fusion_runner.py
    )
    set_tests_properties(Benchmark.AttentionLayoutFusionRunnerContract PROPERTIES
        LABELS "cpu;benchmark;training;attention;schema")
    add_test(
        NAME Benchmark.HipGraphMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_hip_graph_matrix.py
    )
    set_tests_properties(Benchmark.HipGraphMatrixContract PROPERTIES
        LABELS "cpu;benchmark;graph;schema")
    add_test(
        NAME Benchmark.HipGraphGemmMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_hip_graph_gemm_matrix.py
    )
    set_tests_properties(Benchmark.HipGraphGemmMatrixContract PROPERTIES
        LABELS "cpu;benchmark;graph;matmul;schema")
    add_test(
        NAME Benchmark.DeferredDeallocationMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_deferred_deallocation_matrix.py
    )
    set_tests_properties(Benchmark.DeferredDeallocationMatrixContract PROPERTIES
        LABELS "cpu;benchmark;lifetime;memory;schema")
    add_test(
        NAME Benchmark.ScopedDeferredModelMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_scoped_deferred_model_matrix.py
    )
    set_tests_properties(Benchmark.ScopedDeferredModelMatrixContract PROPERTIES
        LABELS "cpu;benchmark;lifetime;model;schema")
    add_test(
        NAME Benchmark.PerDeviceHandleRegressionContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_per_device_handle_regression.py
    )
    set_tests_properties(Benchmark.PerDeviceHandleRegressionContract PROPERTIES
        LABELS "cpu;benchmark;multi_gpu;matmul;schema")
    add_test(
        NAME Benchmark.StreamOrderedAllocatorMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_stream_ordered_allocator_matrix.py
    )
    set_tests_properties(Benchmark.StreamOrderedAllocatorMatrixContract PROPERTIES
        LABELS "cpu;benchmark;lifetime;memory;graph;schema")
    add_test(
        NAME Benchmark.ActivationArenaMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_activation_arena_matrix.py
    )
    set_tests_properties(Benchmark.ActivationArenaMatrixContract PROPERTIES
        LABELS "cpu;benchmark;lifetime;memory;graph;schema")
    add_test(
        NAME Benchmark.ArenaFfnMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_arena_ffn_matrix.py
    )
    set_tests_properties(Benchmark.ArenaFfnMatrixContract PROPERTIES
        LABELS "cpu;benchmark;model;ffn;memory;graph;schema")
    add_test(
        NAME Benchmark.Bf16ArenaFfnMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_arena_ffn_matrix.py
    )
    set_tests_properties(Benchmark.Bf16ArenaFfnMatrixContract PROPERTIES
        LABELS "cpu;benchmark;model;ffn;bf16;memory;graph;schema")
    add_test(
        NAME Benchmark.Bf16FfnArenaModelsContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_ffn_arena_models.py
    )
    set_tests_properties(Benchmark.Bf16FfnArenaModelsContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;ffn;bf16;memory;schema")
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
        inference/hip_shape_matrix_test.cpp
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
