add_executable(microllm_bench_ops micro/benchmark_ops.cpp)
target_link_libraries(microllm_bench_ops PRIVATE microllm::ops microllm::runtime)

if(MICROLLM_HAS_HIP AND MICROLLM_HAS_HIPBLASLT)
    add_executable(microllm_bench_precision micro/benchmark_precision.cpp)
    target_link_libraries(microllm_bench_precision PRIVATE microllm::ops microllm::runtime)
endif()

add_executable(microllm_bench_model end_to_end/benchmark_model.cpp)
target_link_libraries(microllm_bench_model PRIVATE
    microllm::training microllm::inference microllm::runtime
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
            foreach(MICROLLM_LOW_DTYPE IN ITEMS fp16 bf16)
                add_test(NAME Benchmark.HipMatmul.${MICROLLM_LOW_DTYPE}
                         COMMAND microllm_bench_ops --op matmul --size 128
                                 --dtype ${MICROLLM_LOW_DTYPE}
                                 --implementation hipblaslt --warmup 1 --repetitions 2
                                 --device hip)
                set_tests_properties(Benchmark.HipMatmul.${MICROLLM_LOW_DTYPE}
                                     PROPERTIES LABELS "hip;benchmark;dtype;smoke")
            endforeach()
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
