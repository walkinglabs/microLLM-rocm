add_executable(microllm_n0_ppm n0_ppm.cpp)
target_link_libraries(microllm_n0_ppm PRIVATE microllm::core)

if(BUILD_TESTING)
    add_test(NAME N0.PpmSmoke COMMAND microllm_n0_ppm "${CMAKE_CURRENT_BINARY_DIR}/n0.ppm")
    set_tests_properties(N0.PpmSmoke PROPERTIES LABELS "cpu;example;smoke")
endif()
