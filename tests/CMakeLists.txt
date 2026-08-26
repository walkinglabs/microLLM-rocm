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

add_executable(microllm_hf_cli_fixture
    io/hf_cli_fixture.cpp
)
target_link_libraries(microllm_hf_cli_fixture PRIVATE
    microllm::model microllm::io)

find_package(Python3 COMPONENTS Interpreter QUIET)

add_test(
    NAME PackageConfig.InstalledConsumer
    COMMAND ${CMAKE_COMMAND}
        -DMICROLLM_BINARY_DIR=${PROJECT_BINARY_DIR}
        -DMICROLLM_CONSUMER_SOURCE_DIR=${PROJECT_SOURCE_DIR}/tests/package/consumer
        -DMICROLLM_CONSUMER_BINARY_DIR=${PROJECT_BINARY_DIR}/package-consumer-build
        -DMICROLLM_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/package-consumer-prefix
        -DMICROLLM_PACKAGE_WITH_CAPI=${MICROLLM_BUILD_CAPI}
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
        -DMICROLLM_PACKAGE_WITH_CAPI=${MICROLLM_BUILD_CAPI}
        -P ${PROJECT_SOURCE_DIR}/cmake/RunBuildTreeConsumer.cmake
)
set_tests_properties(PackageConfig.BuildTreeConsumer PROPERTIES
    LABELS "${microllm_package_test_labels}"
    RUN_SERIAL TRUE
)

add_test(
    NAME PackageConfig.PublicExample
    COMMAND ${CMAKE_COMMAND}
        -DMICROLLM_BINARY_DIR=${PROJECT_BINARY_DIR}
        -DMICROLLM_EXAMPLE_SOURCE_DIR=${PROJECT_SOURCE_DIR}/examples/package-consumer
        -DMICROLLM_EXAMPLE_BINARY_DIR=${PROJECT_BINARY_DIR}/package-public-example-build
        -DMICROLLM_EXAMPLE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/package-public-example-prefix
        -P ${PROJECT_SOURCE_DIR}/cmake/RunPackageExample.cmake
)
set_tests_properties(PackageConfig.PublicExample PROPERTIES
    LABELS "${microllm_package_test_labels};example"
    RUN_SERIAL TRUE
)

add_test(
    NAME PackageConfig.RejectsNonRelocatableDestination
    COMMAND ${CMAKE_COMMAND}
        -DMICROLLM_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -DMICROLLM_BINARY_DIR=${PROJECT_BINARY_DIR}
        -P ${PROJECT_SOURCE_DIR}/cmake/RunInvalidPackageDestination.cmake
)
set_tests_properties(PackageConfig.RejectsNonRelocatableDestination PROPERTIES
    LABELS "cpu;package;configure"
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
        NAME HfCLI.BatchLogitsExport
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_hf_cli_batch_logits.py
                --binary $<TARGET_FILE:microllm_hf_infer>
                --fixture-helper $<TARGET_FILE:microllm_hf_cli_fixture>
    )
    set_tests_properties(HfCLI.BatchLogitsExport PROPERTIES
        LABELS "cpu;cli;binary;batch;inference;integration")
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
        NAME Benchmark.AdamWMomentMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_adamw_moment_matrix.py
    )
    set_tests_properties(Benchmark.AdamWMomentMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;optimizer;precision;memory;schema")
    add_test(
        NAME Benchmark.TrainingProfileDeltaContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_profile_step_delta.py
    )
    set_tests_properties(Benchmark.TrainingProfileDeltaContract PROPERTIES
        LABELS "cpu;benchmark;training;profile;schema")
    add_test(
        NAME Benchmark.GroupedWeightGradientMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_grouped_weight_gradient_matrix.py
    )
    set_tests_properties(Benchmark.GroupedWeightGradientMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;matmul;grouped;schema")
    add_test(
        NAME Benchmark.PackedWeightGradientMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_packed_weight_gradient_matrix.py
    )
    set_tests_properties(Benchmark.PackedWeightGradientMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;matmul;memory;schema")
    add_test(
        NAME Benchmark.Fp32WeightGradientSolutionMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_fp32_weight_gradient_solution_matrix.py
    )
    set_tests_properties(Benchmark.Fp32WeightGradientSolutionMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;matmul;solution;schema")
    add_test(
        NAME Benchmark.Fp32WeightGradientSolutionModelsContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_fp32_weight_gradient_solution_models.py
    )
    set_tests_properties(Benchmark.Fp32WeightGradientSolutionModelsContract PROPERTIES
        LABELS "cpu;benchmark;training;matmul;solution;model;schema")
    add_test(
        NAME Benchmark.TrainingGraphCaptureMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_training_graph_capture_matrix.py
    )
    set_tests_properties(Benchmark.TrainingGraphCaptureMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;graph;lifetime;schema")
    add_test(
        NAME Benchmark.AdamWGraphReplayMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_adamw_graph_replay_matrix.py
    )
    set_tests_properties(Benchmark.AdamWGraphReplayMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;graph;optimizer;schema")
    add_test(
        NAME Benchmark.AdamWGraphMultiMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_adamw_graph_multi_matrix.py
    )
    set_tests_properties(Benchmark.AdamWGraphMultiMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;graph;optimizer;memory;schema")
    add_test(
        NAME Benchmark.GradientAddressMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_gradient_address_matrix.py
    )
    set_tests_properties(Benchmark.GradientAddressMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;graph;autograd;memory;schema")
    add_test(
        NAME Benchmark.OptimizerGraphModelPreflightContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_optimizer_graph_model_preflight.py
    )
    set_tests_properties(Benchmark.OptimizerGraphModelPreflightContract PROPERTIES
        LABELS "cpu;benchmark;training;graph;optimizer;allocator;schema")
    add_test(
        NAME Benchmark.QuiescentAllocatorHandoffMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_quiescent_allocator_handoff_matrix.py
    )
    set_tests_properties(Benchmark.QuiescentAllocatorHandoffMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;graph;allocator;lifetime;schema")
    add_test(
        NAME Benchmark.OptimizerGraphModelMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_optimizer_graph_model_matrix.py
    )
    set_tests_properties(Benchmark.OptimizerGraphModelMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;graph;optimizer;model;schema")
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
        NAME Benchmark.HfAllocationSourcesContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_hf_allocation_sources.py
    )
    set_tests_properties(Benchmark.HfAllocationSourcesContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;memory;profile;schema")
    add_test(
        NAME Benchmark.Fp32AttentionSolutionMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_fp32_attention_solution_matrix.py
    )
    set_tests_properties(Benchmark.Fp32AttentionSolutionMatrixContract PROPERTIES
        LABELS "cpu;benchmark;attention;matmul;solution;schema")
    add_test(
        NAME Benchmark.Fp32AttentionSolutionModelsContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_fp32_attention_solution_models.py
    )
    set_tests_properties(Benchmark.Fp32AttentionSolutionModelsContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;attention;matmul;solution;schema")
    add_test(
        NAME Benchmark.Bf16SwiGLUVectorMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_swiglu_vector_matrix.py
    )
    set_tests_properties(Benchmark.Bf16SwiGLUVectorMatrixContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;ffn;bf16;schema")
    add_test(
        NAME Benchmark.Bf16SwiGLUModelsContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_swiglu_models.py
    )
    set_tests_properties(Benchmark.Bf16SwiGLUModelsContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;ffn;bf16;schema")
    add_test(
        NAME Benchmark.Bf16RmsNormOutputMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_rms_norm_output_matrix.py
    )
    set_tests_properties(Benchmark.Bf16RmsNormOutputMatrixContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;norm;bf16;schema")
    add_test(
        NAME Benchmark.Bf16GroupedQkvMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_grouped_qkv_matrix.py
    )
    set_tests_properties(Benchmark.Bf16GroupedQkvMatrixContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;bf16;matmul;grouped;schema")
    add_test(
        NAME Benchmark.Bf16GroupedQkvModelsContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_grouped_qkv_models.py
    )
    set_tests_properties(Benchmark.Bf16GroupedQkvModelsContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;bf16;matmul;grouped;memory;schema")
    add_test(
        NAME Benchmark.Bf16GroupedQkvPrewarmContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_grouped_qkv_prewarm.py
    )
    set_tests_properties(Benchmark.Bf16GroupedQkvPrewarmContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;bf16;grouped;latency;schema")
    add_test(
        NAME Benchmark.HipblasltPreloadContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_hipblaslt_preload.py
    )
    set_tests_properties(Benchmark.HipblasltPreloadContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;bf16;matmul;latency;schema")
    add_test(
        NAME Benchmark.Bf16ExactStartupContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_exact_startup.py
    )
    set_tests_properties(Benchmark.Bf16ExactStartupContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;bf16;matmul;solution;latency;schema")
    add_test(
        NAME Benchmark.Bf16GroupedGateUpMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_grouped_gate_up_matrix.py
    )
    set_tests_properties(Benchmark.Bf16GroupedGateUpMatrixContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;ffn;bf16;matmul;grouped;schema")
    add_test(
        NAME Benchmark.Bf16GroupedGateUpModelsContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_grouped_gate_up_models.py
    )
    set_tests_properties(Benchmark.Bf16GroupedGateUpModelsContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;ffn;bf16;matmul;grouped;memory;schema")
    add_test(
        NAME Benchmark.Bf16GroupedCompositionContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_grouped_composition.py
    )
    set_tests_properties(Benchmark.Bf16GroupedCompositionContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;attention;ffn;bf16;matmul;grouped;memory;schema")
    add_test(
        NAME Benchmark.Bf16GroupedShapeMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_grouped_shape_matrix.py
    )
    set_tests_properties(Benchmark.Bf16GroupedShapeMatrixContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;attention;ffn;bf16;matmul;grouped;shape;schema")
    add_test(
        NAME Benchmark.Bf16GroupedShapeModelsContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_grouped_shape_models.py
    )
    set_tests_properties(Benchmark.Bf16GroupedShapeModelsContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;attention;ffn;bf16;matmul;grouped;shape;batch;memory;schema")
    add_test(
        NAME Benchmark.HfStridedCopySourcesContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_hf_strided_copy_sources.py
    )
    set_tests_properties(Benchmark.HfStridedCopySourcesContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;attention;memory;profile;schema")
    add_test(
        NAME Benchmark.InferenceBthdAttentionContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_inference_bthd_attention.py
    )
    set_tests_properties(Benchmark.InferenceBthdAttentionContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;attention;bf16;memory;layout;schema")
    add_test(
        NAME Benchmark.InferenceBthdBf16QkContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_inference_bthd_bf16_qk.py
    )
    set_tests_properties(Benchmark.InferenceBthdBf16QkContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;attention;bf16;matmul;grouped;memory;layout;schema")
    add_test(
        NAME Benchmark.InferenceBthdBf16QkShapesContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_inference_bthd_bf16_qk_shapes.py
    )
    set_tests_properties(Benchmark.InferenceBthdBf16QkShapesContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;attention;bf16;matmul;grouped;memory;layout;shape;batch;schema")
    add_test(
        NAME Benchmark.CausalSoftmaxThreadsContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_causal_softmax_threads.py
    )
    set_tests_properties(Benchmark.CausalSoftmaxThreadsContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;attention;softmax;memory;shape;schema")
    add_test(
        NAME Benchmark.RocwmmaQkMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_rocwmma_qk_matrix.py
    )
    set_tests_properties(Benchmark.RocwmmaQkMatrixContract PROPERTIES
        LABELS "cpu;benchmark;inference;attention;bf16;mfma;rocwmma;shape;schema")
    add_test(
        NAME Benchmark.RocwmmaOnlineAttentionMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_rocwmma_online_attention_matrix.py
    )
    set_tests_properties(Benchmark.RocwmmaOnlineAttentionMatrixContract PROPERTIES
        LABELS "cpu;benchmark;inference;attention;bf16;mfma;rocwmma;online;gqa;memory;shape;schema")
    add_test(
        NAME Benchmark.RocwmmaOnlineOperatorMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_rocwmma_online_operator_matrix.py
    )
    set_tests_properties(Benchmark.RocwmmaOnlineOperatorMatrixContract PROPERTIES
        LABELS "cpu;benchmark;inference;attention;bf16;mfma;rocwmma;online;gqa;memory;shape;batch;schema")
    add_test(
        NAME Benchmark.RocwmmaOnlineModelMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_rocwmma_online_model_matrix.py
    )
    set_tests_properties(Benchmark.RocwmmaOnlineModelMatrixContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;attention;bf16;mfma;rocwmma;online;gqa;memory;shape;batch;schema")
    add_test(
        NAME Benchmark.CurrentInferenceProfileContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_profile_current_inference.py
    )
    set_tests_properties(Benchmark.CurrentInferenceProfileContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;profile;rocprof;schema")
    add_test(
        NAME Benchmark.CurrentCachedDecodeProfileContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_profile_current_cached_decode.py
    )
    set_tests_properties(Benchmark.CurrentCachedDecodeProfileContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;profile;cache;rocprof;schema")
    add_test(
        NAME Benchmark.CachedAttentionStageMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_cached_attention_stage_matrix.py
    )
    set_tests_properties(Benchmark.CachedAttentionStageMatrixContract PROPERTIES
        LABELS "cpu;benchmark;inference;attention;cache;precision;schema")
    add_test(
        NAME Benchmark.CachedAttentionSplitModelContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_cached_attention_split_models.py
    )
    set_tests_properties(Benchmark.CachedAttentionSplitModelContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;attention;cache;precision;schema")
    add_test(
        NAME Benchmark.CurrentTrainingProfileContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_profile_current_training.py
    )
    set_tests_properties(Benchmark.CurrentTrainingProfileContract PROPERTIES
        LABELS "cpu;benchmark;model;training;profile;rocprof;schema")
    add_test(
        NAME Benchmark.Bf16WeightGradientMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_weight_gradient_matrix.py
    )
    set_tests_properties(Benchmark.Bf16WeightGradientMatrixContract PROPERTIES
        LABELS "cpu;benchmark;model;training;backward;matmul;bf16;shape;schema")
    add_test(
        NAME Benchmark.TrainingTrajectoryEvidenceContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_training_trajectory_evidence.py
    )
    set_tests_properties(Benchmark.TrainingTrajectoryEvidenceContract PROPERTIES
        LABELS "cpu;benchmark;model;training;precision;trajectory;weights;schema")
    add_test(
        NAME Benchmark.Bf16WeightGradientWorkspaceMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_weight_gradient_workspace_matrix.py
    )
    set_tests_properties(Benchmark.Bf16WeightGradientWorkspaceMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;backward;bf16;memory;workspace;schema")
    add_test(
        NAME Benchmark.GradientProducerOutMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_gradient_producer_out_matrix.py
    )
    set_tests_properties(Benchmark.GradientProducerOutMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;gradient;memory;schema")
    add_test(
        NAME Benchmark.ExternalGradientPoolMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_external_gradient_pool_matrix.py
    )
    set_tests_properties(Benchmark.ExternalGradientPoolMatrixContract PROPERTIES
        LABELS "cpu;benchmark;autograd;training;gradient;memory;model;schema")
    add_test(
        NAME Benchmark.PyTorchRocmCustomOpMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_pytorch_rocm_custom_op_matrix.py
    )
    set_tests_properties(Benchmark.PyTorchRocmCustomOpMatrixContract PROPERTIES
        LABELS "cpu;benchmark;torch;dtype;autograd;profile;memory;schema")
    add_test(
        NAME Benchmark.PyTorchCustomOpVector16Contract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_pytorch_custom_op_vector16.py
    )
    set_tests_properties(Benchmark.PyTorchCustomOpVector16Contract PROPERTIES
        LABELS "cpu;benchmark;torch;dtype;performance;memory;schema")
    add_test(
        NAME Benchmark.PyTorchCustomOpSwiGLUContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_pytorch_custom_op_swiglu.py
    )
    set_tests_properties(Benchmark.PyTorchCustomOpSwiGLUContract PROPERTIES
        LABELS "cpu;benchmark;torch;dtype;autograd;performance;memory;schema")
    add_test(
        NAME Benchmark.PyTorchSwiGLUBackwardResultContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_pytorch_swiglu_backward_result.py
    )
    set_tests_properties(Benchmark.PyTorchSwiGLUBackwardResultContract PROPERTIES
        LABELS "cpu;benchmark;torch;autograd;backward;performance;memory;schema")
    add_test(
        NAME Benchmark.PyTorchSwiGLUScalarSeedContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_pytorch_swiglu_scalar_seed.py
    )
    set_tests_properties(Benchmark.PyTorchSwiGLUScalarSeedContract PROPERTIES
        LABELS "cpu;benchmark;torch;autograd;backward;performance;memory;schema")
    add_test(
        NAME Benchmark.PyTorchSwiGLUAutogradAttributionContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_pytorch_swiglu_autograd_attribution.py
    )
    set_tests_properties(Benchmark.PyTorchSwiGLUAutogradAttributionContract PROPERTIES
        LABELS "cpu;benchmark;torch;autograd;backward;profile;performance;schema")
    add_test(
        NAME Benchmark.PyTorchSwiGLUCompileResultContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_pytorch_swiglu_compile_result.py
    )
    set_tests_properties(Benchmark.PyTorchSwiGLUCompileResultContract PROPERTIES
        LABELS "cpu;benchmark;torch;autograd;compile;profile;performance;schema")
    add_test(
        NAME Benchmark.PyTorchSwiGLUCppAutogradContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_pytorch_swiglu_cpp_autograd.py
    )
    set_tests_properties(Benchmark.PyTorchSwiGLUCppAutogradContract PROPERTIES
        LABELS "cpu;benchmark;torch;autograd;backward;performance;memory;schema")
    add_test(
        NAME Benchmark.PyTorchSwiGLUTypedBackwardContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_pytorch_swiglu_typed_backward.py
    )
    set_tests_properties(Benchmark.PyTorchSwiGLUTypedBackwardContract PROPERTIES
        LABELS "cpu;benchmark;torch;dtype;autograd;backward;performance;memory;schema")
    add_test(
        NAME Benchmark.PyTorchTypedSoftmaxContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_pytorch_typed_softmax.py
    )
    set_tests_properties(Benchmark.PyTorchTypedSoftmaxContract PROPERTIES
        LABELS "cpu;benchmark;torch;dtype;softmax;performance;memory;schema")
    add_test(
        NAME Benchmark.PyTorchBlockSoftmaxContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_pytorch_block_softmax.py
    )
    set_tests_properties(Benchmark.PyTorchBlockSoftmaxContract PROPERTIES
        LABELS "cpu;benchmark;torch;dtype;softmax;performance;memory;schema")
    add_test(
        NAME Benchmark.DataParallelVerificationMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_data_parallel_verification_matrix.py
    )
    set_tests_properties(Benchmark.DataParallelVerificationMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;multi_gpu;rccl;verification;schema")
    add_test(
        NAME Benchmark.DataParallelBucketMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_data_parallel_bucket_matrix.py
    )
    set_tests_properties(Benchmark.DataParallelBucketMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;multi_gpu;rccl;bucket;schema")
    add_test(
        NAME Benchmark.DistributedModelWorkloadContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_distributed_model_workload.py
    )
    set_tests_properties(Benchmark.DistributedModelWorkloadContract PROPERTIES
        LABELS "cpu;benchmark;training;model;multi_gpu;rccl;memory;schema")
    add_test(
        NAME Benchmark.DataParallelModelSBucketMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_data_parallel_model_s_bucket_matrix.py
    )
    set_tests_properties(Benchmark.DataParallelModelSBucketMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;model;multi_gpu;rccl;bucket;memory;schema")
    add_test(
        NAME Benchmark.DataParallelInplaceAverageMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_data_parallel_inplace_average_matrix.py
    )
    set_tests_properties(Benchmark.DataParallelInplaceAverageMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;model;multi_gpu;rccl;bucket;memory;schema")
    add_test(
        NAME Benchmark.DataParallelPersistentBucketMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_data_parallel_persistent_bucket_matrix.py
    )
    set_tests_properties(Benchmark.DataParallelPersistentBucketMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;model;multi_gpu;rccl;bucket;memory;schema")
    add_test(
        NAME Benchmark.DataParallelGradientViewMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_data_parallel_gradient_view_matrix.py
    )
    set_tests_properties(Benchmark.DataParallelGradientViewMatrixContract PROPERTIES
        LABELS "cpu;benchmark;training;model;multi_gpu;rccl;bucket;memory;schema")
    add_test(
        NAME Benchmark.DataParallelGradientReadyAuditContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_data_parallel_gradient_ready_audit.py
    )
    set_tests_properties(Benchmark.DataParallelGradientReadyAuditContract PROPERTIES
        LABELS "cpu;benchmark;autograd;training;model;multi_gpu;rccl;bucket;schema")
    add_test(
        NAME Benchmark.DataParallelGradientOverlapMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_data_parallel_gradient_overlap_matrix.py
    )
    set_tests_properties(Benchmark.DataParallelGradientOverlapMatrixContract PROPERTIES
        LABELS "cpu;benchmark;autograd;training;model;multi_gpu;rccl;bucket;memory;schema")
    add_test(
        NAME DistributedRank.LauncherContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_ranked_launcher_contract.py
    )
    set_tests_properties(DistributedRank.LauncherContract PROPERTIES
        LABELS "cpu;multi_gpu;rccl;distributed;failure;schema")
    add_test(
        NAME DistributedRank.MatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_ranked_training_matrix.py
    )
    set_tests_properties(DistributedRank.MatrixContract PROPERTIES
        LABELS "cpu;multi_gpu;rccl;distributed;training;failure;schema")
    add_test(
        NAME DistributedRank.ContextMatrixContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_ranked_overlap_context_matrix.py
    )
    set_tests_properties(DistributedRank.ContextMatrixContract PROPERTIES
        LABELS "cpu;multi_gpu;rccl;distributed;training;context;schema")
    add_test(
        NAME DistributedRank.CheckpointContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_ranked_checkpoint_contract.py
    )
    set_tests_properties(DistributedRank.CheckpointContract PROPERTIES
        LABELS "cpu;multi_gpu;rccl;distributed;training;checkpoint;failure;schema")
    add_test(
        NAME Benchmark.Bf16RepeatContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_bf16_repeat.py
    )
    set_tests_properties(Benchmark.Bf16RepeatContract PROPERTIES
        LABELS "cpu;benchmark;inference;attention;bf16;memory;shape;batch;schema")
    add_test(
        NAME Benchmark.InferenceBthdShapeModelsContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_inference_bthd_shape_models.py
    )
    set_tests_properties(Benchmark.InferenceBthdShapeModelsContract PROPERTIES
        LABELS "cpu;benchmark;model;inference;attention;bf16;memory;layout;shape;batch;schema")
    add_test(
        NAME Coverage.OperatorAndTestFiles
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/audit_test_coverage.py
    )
    set_tests_properties(Coverage.OperatorAndTestFiles PROPERTIES LABELS "cpu;coverage;schema")
    add_test(
        NAME Documentation.StatusContract
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_status_contract.py
    )
    set_tests_properties(Documentation.StatusContract PROPERTIES
        LABELS "cpu;documentation;status;schema")
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
    if(Python3_Interpreter_FOUND AND TARGET microllm_distributed_rank)
        add_test(
            NAME DistributedRank.MultiProcessSmoke
            COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tools/distributed/run_ranked.py
                    --binary $<TARGET_FILE:microllm_distributed_rank>
                    --output-directory ${PROJECT_BINARY_DIR}/ranked-smoke
                    --steps 2 --timeout-seconds 20 --overwrite
        )
        set_tests_properties(DistributedRank.MultiProcessSmoke PROPERTIES
            LABELS "hip;rccl;multi_gpu;distributed;integration"
            RUN_SERIAL TRUE TIMEOUT 30)
        add_test(
            NAME DistributedRank.BucketSmoke
            COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tools/distributed/run_ranked.py
                    --binary $<TARGET_FILE:microllm_distributed_rank>
                    --output-directory ${PROJECT_BINARY_DIR}/ranked-bucket-smoke
                    --steps 2 --timeout-seconds 20 --reducer bucket --overwrite
        )
        set_tests_properties(DistributedRank.BucketSmoke PROPERTIES
            LABELS "hip;rccl;multi_gpu;distributed;integration;bucket"
            RUN_SERIAL TRUE TIMEOUT 30)
        add_test(
            NAME DistributedRank.WeightedOverlapSmoke
            COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tools/distributed/run_ranked.py
                    --binary $<TARGET_FILE:microllm_distributed_rank>
                    --output-directory
                    ${PROJECT_BINARY_DIR}/ranked-weighted-overlap-smoke
                    --steps 3 --timeout-seconds 20
                    --rank-batch-rows 1,2 --input-weighting token-weighted
                    --reducer overlap-views --bucket-bytes 4096 --overwrite
        )
        set_tests_properties(DistributedRank.WeightedOverlapSmoke PROPERTIES
            LABELS "hip;rccl;multi_gpu;distributed;integration;bucket;overlap;weighting"
            RUN_SERIAL TRUE TIMEOUT 30)
        add_test(
            NAME DistributedRank.BucketWeightedOverlapSmoke
            COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tools/distributed/run_ranked.py
                    --binary $<TARGET_FILE:microllm_distributed_rank>
                    --output-directory
                    ${PROJECT_BINARY_DIR}/ranked-bucket-weighted-overlap-smoke
                    --steps 3 --timeout-seconds 20
                    --rank-batch-rows 1,2 --input-weighting token-weighted
                    --reducer bucket-weighted-overlap --bucket-bytes 4096
                    --overwrite
        )
        set_tests_properties(DistributedRank.BucketWeightedOverlapSmoke PROPERTIES
            LABELS "hip;rccl;multi_gpu;distributed;integration;bucket;overlap;weighting"
            RUN_SERIAL TRUE TIMEOUT 30)
        add_test(
            NAME DistributedRank.GatherWeightedOverlapSmoke
            COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tools/distributed/run_ranked.py
                    --binary $<TARGET_FILE:microllm_distributed_rank>
                    --output-directory
                    ${PROJECT_BINARY_DIR}/ranked-gather-weighted-overlap-smoke
                    --steps 3 --timeout-seconds 20
                    --rank-batch-rows 1,2 --input-weighting token-weighted
                    --reducer gather-weighted-overlap --bucket-bytes 4096
                    --overwrite
        )
        set_tests_properties(DistributedRank.GatherWeightedOverlapSmoke PROPERTIES
            LABELS "hip;rccl;multi_gpu;distributed;integration;bucket;overlap;weighting;gather"
            RUN_SERIAL TRUE TIMEOUT 30)
        add_test(
            NAME DistributedRank.PeerFailure
            COMMAND ${Python3_EXECUTABLE}
                    ${PROJECT_SOURCE_DIR}/tools/distributed/run_ranked.py
                    --binary $<TARGET_FILE:microllm_distributed_rank>
                    --output-directory ${PROJECT_BINARY_DIR}/ranked-peer-failure
                    --steps 1 --timeout-seconds 5 --failure-mode peer-failure
                    --overwrite
        )
        set_tests_properties(DistributedRank.PeerFailure PROPERTIES
            LABELS "hip;rccl;multi_gpu;distributed;failure"
            RUN_SERIAL TRUE TIMEOUT 15)
    endif()
endif()
