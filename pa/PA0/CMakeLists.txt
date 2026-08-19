add_executable(microllm_pa0_linear linear_update.cpp)
target_link_libraries(microllm_pa0_linear PRIVATE microllm_project_options)

if(BUILD_TESTING)
    add_test(NAME PA0.LinearUpdate COMMAND microllm_pa0_linear)
    set_tests_properties(PA0.LinearUpdate PROPERTIES LABELS "cpu;pa;smoke")
endif()
