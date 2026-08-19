add_executable(microllm_n0_ppm n0_ppm.cpp)
target_link_libraries(microllm_n0_ppm PRIVATE microllm::core)

add_executable(microllm_n1_cpu_hip n1_cpu_hip.cpp)
target_link_libraries(microllm_n1_cpu_hip PRIVATE microllm::ops microllm::runtime)

add_executable(microllm_model_info model_info.cpp)
target_link_libraries(microllm_model_info PRIVATE microllm::model)

add_executable(microllm_tiny_overfit tiny_overfit.cpp)
target_link_libraries(microllm_tiny_overfit PRIVATE microllm::training microllm::inference)

if(BUILD_TESTING)
    add_test(NAME N0.PpmSmoke COMMAND microllm_n0_ppm "${CMAKE_CURRENT_BINARY_DIR}/n0.ppm")
    set_tests_properties(N0.PpmSmoke PROPERTIES LABELS "cpu;example;smoke")
    if(MICROLLM_HAS_HIP)
        add_test(NAME N1.CpuHipCompare COMMAND microllm_n1_cpu_hip)
        set_tests_properties(N1.CpuHipCompare PROPERTIES LABELS "hip;example;smoke")
    endif()
    add_test(NAME N4.TinyOverfit COMMAND microllm_tiny_overfit)
    set_tests_properties(N4.TinyOverfit PROPERTIES LABELS "cpu;model;smoke")
endif()
