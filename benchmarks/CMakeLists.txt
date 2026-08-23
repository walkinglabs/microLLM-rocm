add_executable(microllm_bench_ops micro/benchmark_ops.cpp)
target_link_libraries(microllm_bench_ops PRIVATE microllm::ops microllm::runtime)

if(MICROLLM_HAS_HIP AND MICROLLM_HAS_HIPBLASLT)
    add_executable(microllm_bench_precision micro/benchmark_precision.cpp)
    target_link_libraries(microllm_bench_precision PRIVATE microllm::ops microllm::runtime)
    add_executable(microllm_bench_int8 micro/benchmark_int8.cpp)
    target_link_libraries(microllm_bench_int8 PRIVATE hip::host roc::hipblaslt)
    add_executable(microllm_bench_bf16_ffn micro/benchmark_bf16_ffn.cpp)
    target_link_libraries(microllm_bench_bf16_ffn PRIVATE
        microllm::ops microllm::runtime hip::host)
    add_executable(microllm_bench_adamw micro/benchmark_adamw.cpp)
    target_link_libraries(microllm_bench_adamw PRIVATE
        microllm::ops microllm::runtime hip::host)
    add_executable(microllm_bench_bf16_algorithms
                   micro/benchmark_bf16_algorithms.cpp)
    target_link_libraries(microllm_bench_bf16_algorithms PRIVATE
        microllm::ops microllm::runtime hip::host roc::hipblaslt)
endif()

add_executable(microllm_bench_model end_to_end/benchmark_model.cpp)
target_link_libraries(microllm_bench_model PRIVATE
    microllm::training microllm::inference microllm::runtime
)
add_executable(microllm_bench_scheduler end_to_end/benchmark_scheduler.cpp)
target_link_libraries(microllm_bench_scheduler PRIVATE
    microllm::inference microllm::runtime
)

if(MICROLLM_HAS_RCCL)
    add_executable(microllm_bench_allreduce distributed/benchmark_allreduce.cpp)
    target_link_libraries(microllm_bench_allreduce PRIVATE
        microllm::multi_gpu microllm::ops microllm::runtime
    )
    add_executable(microllm_bench_overlap distributed/benchmark_overlap.cpp)
    target_link_libraries(microllm_bench_overlap PRIVATE
        microllm::multi_gpu microllm::ops microllm::runtime
    )
endif()

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    add_test(NAME Benchmark.AddSmoke
             COMMAND microllm_bench_ops --op add --size 64 --warmup 1 --repetitions 2 --device cpu)
    set_tests_properties(Benchmark.AddSmoke PROPERTIES LABELS "cpu;benchmark;smoke")
    add_test(NAME Benchmark.ModelTrainSmoke
             COMMAND microllm_bench_model --mode train --model tiny --device cpu
                     --steps 2 --warmup 1 --batch 1 --context 4 --new-tokens 4)
    set_tests_properties(Benchmark.ModelTrainSmoke PROPERTIES LABELS "cpu;benchmark;model;smoke")
    add_test(NAME Benchmark.SchedulerReferenceSmoke
             COMMAND microllm_bench_scheduler --device cpu --requests 3
                     --warmup 0 --repetitions 1 --continuous-slots 2)
    set_tests_properties(Benchmark.SchedulerReferenceSmoke PROPERTIES
                         LABELS "cpu;benchmark;inference;scheduler;schema")
    add_test(NAME Benchmark.SchedulerContinuousProfileSmoke
             COMMAND microllm_bench_scheduler --device cpu --requests 3
                     --warmup 0 --repetitions 1 --continuous-slots 2
                     --continuous-only true)
    set_tests_properties(Benchmark.SchedulerContinuousProfileSmoke PROPERTIES
                         LABELS "cpu;benchmark;inference;scheduler;profile;schema")
    if(Python3_Interpreter_FOUND)
        add_test(NAME Benchmark.JsonSchema
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/validate_json.py
                         $<TARGET_FILE:microllm_bench_ops> $<TARGET_FILE:microllm_bench_model>)
        set_tests_properties(Benchmark.JsonSchema PROPERTIES LABELS "cpu;benchmark;schema")
        add_test(NAME Benchmark.ModelMatrixSchema
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/single_gpu/model_matrix.py
                         --benchmark $<TARGET_FILE:microllm_bench_model>
                         --device cpu --profiles tiny --modes train,generate
                         --output ${CMAKE_CURRENT_BINARY_DIR}/cpu-model-matrix.jsonl)
        set_tests_properties(Benchmark.ModelMatrixSchema
                             PROPERTIES LABELS "cpu;benchmark;model;memory;schema")
        add_test(NAME Benchmark.HfModelMatrixUnavailableSchema
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/single_gpu/hf_model_matrix.py
                         --manifest
                         ${CMAKE_CURRENT_SOURCE_DIR}/single_gpu/hf_models.example.json
                         --infer-binary $<TARGET_FILE:microllm_hf_infer>
                         --train-binary $<TARGET_FILE:microllm_hf_train_step>
                         --device cpu --modes infer,train --allow-unavailable
                         --output ${CMAKE_CURRENT_BINARY_DIR}/hf-unavailable-matrix.jsonl)
        set_tests_properties(Benchmark.HfModelMatrixUnavailableSchema PROPERTIES
                             LABELS "cpu;benchmark;huggingface;memory;schema")
        add_test(NAME Benchmark.HfBatchContract
                 COMMAND ${Python3_EXECUTABLE}
                         ${PROJECT_SOURCE_DIR}/python/tests/test_hf_benchmark_contract.py)
        set_tests_properties(Benchmark.HfBatchContract PROPERTIES
                             LABELS "cpu;benchmark;huggingface;batch;schema")
        add_test(NAME Benchmark.HfShapeMatrixContract
                 COMMAND ${Python3_EXECUTABLE}
                         ${PROJECT_SOURCE_DIR}/python/tests/test_hf_shape_matrix.py)
        set_tests_properties(Benchmark.HfShapeMatrixContract PROPERTIES
                             LABELS "cpu;benchmark;huggingface;batch;context;schema")
        add_test(NAME Benchmark.HfInferenceShapeMatrixContract
                 COMMAND ${Python3_EXECUTABLE}
                         ${PROJECT_SOURCE_DIR}/python/tests/test_hf_inference_shape_matrix.py)
        set_tests_properties(Benchmark.HfInferenceShapeMatrixContract PROPERTIES
                             LABELS "cpu;benchmark;huggingface;inference;batch;context;kv-cache;schema")
        add_test(NAME Benchmark.HfContinuousMatrixContract
                 COMMAND ${Python3_EXECUTABLE}
                         ${PROJECT_SOURCE_DIR}/python/tests/test_hf_continuous_matrix.py)
        set_tests_properties(Benchmark.HfContinuousMatrixContract PROPERTIES
                             LABELS "cpu;benchmark;huggingface;inference;scheduler;batch;context;kv-cache;schema")
        add_test(NAME Benchmark.Mi300PrecisionRooflineContract
                 COMMAND ${Python3_EXECUTABLE}
                         ${PROJECT_SOURCE_DIR}/python/tests/test_mi300_precision_roofline.py)
        set_tests_properties(Benchmark.Mi300PrecisionRooflineContract PROPERTIES
                             LABELS "cpu;benchmark;precision;roofline;schema")
        add_test(NAME Benchmark.Mi300Int8ProbeContract
                 COMMAND ${Python3_EXECUTABLE}
                         ${PROJECT_SOURCE_DIR}/python/tests/test_mi300_int8_probe.py)
        set_tests_properties(Benchmark.Mi300Int8ProbeContract PROPERTIES
                             LABELS "cpu;benchmark;precision;int8;roofline;schema")
        add_test(NAME Benchmark.KvCachePrecisionContract
                 COMMAND ${Python3_EXECUTABLE}
                         ${PROJECT_SOURCE_DIR}/python/tests/test_kv_cache_precision.py)
        set_tests_properties(Benchmark.KvCachePrecisionContract PROPERTIES
                             LABELS "cpu;benchmark;huggingface;inference;precision;kv-cache;schema")
        add_test(NAME Benchmark.KvCachePolicyComparisonContract
                 COMMAND ${Python3_EXECUTABLE}
                         ${PROJECT_SOURCE_DIR}/python/tests/test_kv_cache_policy_comparison.py)
        set_tests_properties(Benchmark.KvCachePolicyComparisonContract PROPERTIES
                             LABELS "cpu;benchmark;huggingface;inference;precision;kv-cache;schema")
        add_test(NAME Benchmark.FrameworkComparisonBuiltinsSchema
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/single_gpu/compare_frameworks.py
                         --kind builtins
                         --microllm
                         ${CMAKE_CURRENT_SOURCE_DIR}/results/2026-08-20-mi300x-pytorch-builtins-comparison/microLLM.jsonl
                         --pytorch
                         ${CMAKE_CURRENT_SOURCE_DIR}/results/2026-08-20-mi300x-pytorch-builtins-comparison/pytorch.jsonl
                         --output ${CMAKE_CURRENT_BINARY_DIR}/framework-builtins-comparison.jsonl)
        add_test(NAME Benchmark.FrameworkComparisonHfSchema
                 COMMAND ${Python3_EXECUTABLE}
                         ${CMAKE_CURRENT_SOURCE_DIR}/single_gpu/compare_frameworks.py
                         --kind huggingface
                         --microllm
                         ${CMAKE_CURRENT_SOURCE_DIR}/results/2026-08-20-mi300x-pytorch-hf-comparison/microLLM.jsonl
                         --pytorch
                         ${CMAKE_CURRENT_SOURCE_DIR}/results/2026-08-20-mi300x-pytorch-hf-comparison/pytorch.jsonl
                         --output ${CMAKE_CURRENT_BINARY_DIR}/framework-hf-comparison.jsonl)
        set_tests_properties(
            Benchmark.FrameworkComparisonBuiltinsSchema
            Benchmark.FrameworkComparisonHfSchema
            PROPERTIES LABELS "cpu;benchmark;pytorch;comparison;schema")
        add_test(NAME OptimizationLog.Validate
                 COMMAND ${Python3_EXECUTABLE}
                         ${PROJECT_SOURCE_DIR}/docs/optimization-log/scripts/validate_log.py)
        set_tests_properties(OptimizationLog.Validate PROPERTIES
                             LABELS "cpu;documentation;optimization;schema")
    endif()
    if(MICROLLM_HAS_HIP)
        add_test(NAME Benchmark.HipAddSmoke
                 COMMAND microllm_bench_ops --op add --size 1024 --warmup 1 --repetitions 2 --device hip)
        set_tests_properties(Benchmark.HipAddSmoke PROPERTIES LABELS "hip;benchmark;smoke")
        if(MICROLLM_HAS_HIPBLASLT)
            add_test(NAME Benchmark.HipPrecisionMatrix
                     COMMAND microllm_bench_precision --size 128 --warmup 1 --repetitions 2)
            set_tests_properties(Benchmark.HipPrecisionMatrix
                                 PROPERTIES LABELS "hip;benchmark;dtype;fp8;smoke")
            add_test(NAME Benchmark.HipInt8Matrix
                     COMMAND microllm_bench_int8 --size 128 --warmup 1 --repetitions 2)
            set_tests_properties(Benchmark.HipInt8Matrix
                                 PROPERTIES LABELS "hip;benchmark;dtype;int8;smoke")
            add_test(NAME Benchmark.HipBf16FfnIsland
                     COMMAND microllm_bench_bf16_ffn --path island --tokens 1
                             --hidden 128 --intermediate 256 --warmup 1 --repetitions 2)
            set_tests_properties(Benchmark.HipBf16FfnIsland
                                 PROPERTIES LABELS "hip;benchmark;dtype;bf16;smoke")
            add_test(NAME Benchmark.HipAdamWVectorized
                     COMMAND microllm_bench_adamw --elements 4099 --mirror true
                             --implementation vectorized --warmup 1 --repetitions 2)
            set_tests_properties(Benchmark.HipAdamWVectorized
                                 PROPERTIES LABELS "hip;benchmark;optimizer;smoke")
            foreach(MICROLLM_LOW_DTYPE IN ITEMS fp16 bf16)
                add_test(NAME Benchmark.HipMatmul.${MICROLLM_LOW_DTYPE}
                         COMMAND microllm_bench_ops --op matmul --size 128
                                 --dtype ${MICROLLM_LOW_DTYPE}
                                 --implementation hipblaslt --warmup 1 --repetitions 2
                                 --device hip)
                set_tests_properties(Benchmark.HipMatmul.${MICROLLM_LOW_DTYPE}
                                     PROPERTIES LABELS "hip;benchmark;dtype;smoke")
            endforeach()
            add_test(NAME Benchmark.HipTransposeMatmul
                     COMMAND microllm_bench_ops --op matmul --device hip
                             --implementation hipblaslt --dtype fp32
                             --m 128 --k 32 --n 128 --transpose-left true
                             --warmup 1 --repetitions 2)
            set_tests_properties(Benchmark.HipTransposeMatmul PROPERTIES
                                 LABELS "hip;benchmark;matmul;transpose;smoke")
            add_test(NAME Benchmark.HipStridedBatchedMatmul
                     COMMAND microllm_bench_ops --op matmul --device hip
                             --implementation hipblaslt --dtype fp32 --batch 2
                             --m 32 --k 16 --n 24 --warmup 1 --repetitions 2)
            set_tests_properties(Benchmark.HipStridedBatchedMatmul PROPERTIES
                                 LABELS "hip;benchmark;matmul;batch;smoke")
        endif()
        add_test(NAME Benchmark.HipGenerateSmoke
                 COMMAND microllm_bench_model --mode generate --model tiny --device hip
                         --steps 2 --warmup 1 --batch 1 --context 1 --new-tokens 4)
        set_tests_properties(Benchmark.HipGenerateSmoke PROPERTIES LABELS "hip;benchmark;model;smoke")
        if(Python3_Interpreter_FOUND)
            add_test(NAME Benchmark.HipModelMatrix
                     COMMAND ${Python3_EXECUTABLE}
                             ${CMAKE_CURRENT_SOURCE_DIR}/single_gpu/model_matrix.py
                             --benchmark $<TARGET_FILE:microllm_bench_model>
                             --device hip --profiles tiny,model-s,model-m
                             --modes train,generate
                             --output ${CMAKE_CURRENT_BINARY_DIR}/hip-model-matrix.jsonl)
            set_tests_properties(Benchmark.HipModelMatrix PROPERTIES
                                 LABELS "hip;benchmark;model;memory;performance;matrix;slow"
                                 TIMEOUT 900)
        endif()
    endif()
    if(MICROLLM_HAS_RCCL)
        add_test(NAME Benchmark.RcclSmoke
                 COMMAND microllm_bench_allreduce --ranks 2 --total-elements 1024
                         --bucket-elements 1024 --warmup 1 --repetitions 2)
        set_tests_properties(Benchmark.RcclSmoke PROPERTIES
                             LABELS "hip;rccl;multi_gpu;benchmark;smoke")
        add_test(NAME Benchmark.RcclOverlapSmoke
                 COMMAND microllm_bench_overlap --payload-elements 1024
                         --compute-elements 1024 --warmup 1 --repetitions 2)
        set_tests_properties(Benchmark.RcclOverlapSmoke PROPERTIES
                             LABELS "hip;rccl;multi_gpu;benchmark;smoke")
    endif()
endif()
