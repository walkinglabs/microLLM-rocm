add_executable(microllm_bench_ops micro/benchmark_ops.cpp)
target_link_libraries(microllm_bench_ops PRIVATE microllm::ops microllm::runtime)

if(BUILD_TESTING)
    add_test(NAME Benchmark.AddSmoke
             COMMAND microllm_bench_ops --op add --size 64 --warmup 1 --repetitions 2 --device cpu)
    set_tests_properties(Benchmark.AddSmoke PROPERTIES LABELS "cpu;benchmark;smoke")
    if(MICROLLM_HAS_HIP)
        add_test(NAME Benchmark.HipAddSmoke
                 COMMAND microllm_bench_ops --op add --size 1024 --warmup 1 --repetitions 2 --device hip)
        set_tests_properties(Benchmark.HipAddSmoke PROPERTIES LABELS "hip;benchmark;smoke")
    endif()
endif()
