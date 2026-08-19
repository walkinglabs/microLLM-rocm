add_executable(microllm_bench_ops micro/benchmark_ops.cpp)
target_link_libraries(microllm_bench_ops PRIVATE microllm::ops microllm::runtime)

add_executable(microllm_bench_model end_to_end/benchmark_model.cpp)
target_link_libraries(microllm_bench_model PRIVATE
    microllm::training microllm::inference microllm::runtime
)

if(MICROLLM_HAS_RCCL)
    add_executable(microllm_bench_allreduce distributed/benchmark_allreduce.cpp)
    target_link_libraries(microllm_bench_allreduce PRIVATE
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
    endif()
    if(MICROLLM_HAS_HIP)
        add_test(NAME Benchmark.HipAddSmoke
                 COMMAND microllm_bench_ops --op add --size 1024 --warmup 1 --repetitions 2 --device hip)
        set_tests_properties(Benchmark.HipAddSmoke PROPERTIES LABELS "hip;benchmark;smoke")
        add_test(NAME Benchmark.HipGenerateSmoke
                 COMMAND microllm_bench_model --mode generate --model tiny --device hip
                         --steps 2 --warmup 1 --batch 1 --context 1 --new-tokens 4)
        set_tests_properties(Benchmark.HipGenerateSmoke PROPERTIES LABELS "hip;benchmark;model;smoke")
    endif()
    if(MICROLLM_HAS_RCCL)
        add_test(NAME Benchmark.RcclSmoke
                 COMMAND microllm_bench_allreduce --ranks 2 --total-elements 1024
                         --bucket-elements 1024 --warmup 1 --repetitions 2)
        set_tests_properties(Benchmark.RcclSmoke PROPERTIES
                             LABELS "hip;rccl;multi_gpu;benchmark;smoke")
    endif()
endif()
