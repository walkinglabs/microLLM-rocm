add_executable(microllm_n0_ppm n0_ppm.cpp)
target_link_libraries(microllm_n0_ppm PRIVATE microllm::core)

add_executable(microllm_n1_cpu_hip n1_cpu_hip.cpp)
target_link_libraries(microllm_n1_cpu_hip PRIVATE microllm::ops microllm::runtime)

add_executable(microllm_model_info model_info.cpp)
target_link_libraries(microllm_model_info PRIVATE microllm::model)

add_executable(microllm_tiny_overfit tiny_overfit.cpp)
target_link_libraries(microllm_tiny_overfit PRIVATE microllm::training microllm::inference)

add_executable(microllm_model_s_smoke model_s_smoke.cpp)
target_link_libraries(microllm_model_s_smoke PRIVATE microllm::model)

add_executable(microllm_model_s_train_smoke model_s_train_smoke.cpp)
target_link_libraries(microllm_model_s_train_smoke PRIVATE microllm::training)

add_executable(microllm_model_s_hip_smoke model_s_hip_smoke.cpp)
target_link_libraries(microllm_model_s_hip_smoke PRIVATE microllm::model microllm::runtime)

if(BUILD_TESTING)
    add_test(NAME N0.PpmSmoke COMMAND microllm_n0_ppm "${CMAKE_CURRENT_BINARY_DIR}/n0.ppm")
    set_tests_properties(N0.PpmSmoke PROPERTIES LABELS "cpu;example;smoke")
    if(MICROLLM_HAS_HIP)
        add_test(NAME N1.CpuHipCompare COMMAND microllm_n1_cpu_hip)
        set_tests_properties(N1.CpuHipCompare PROPERTIES LABELS "hip;example;smoke")
        add_test(NAME ModelS.HipForwardSmoke COMMAND microllm_model_s_hip_smoke)
        set_tests_properties(ModelS.HipForwardSmoke PROPERTIES LABELS "hip;model;slow")
    endif()
    add_test(NAME N4.TinyOverfit COMMAND microllm_tiny_overfit)
    set_tests_properties(N4.TinyOverfit PROPERTIES LABELS "cpu;model;smoke")
    add_test(NAME ModelS.ForwardSmoke COMMAND microllm_model_s_smoke)
    set_tests_properties(ModelS.ForwardSmoke PROPERTIES LABELS "cpu;model;slow")
    add_test(NAME ModelS.TrainStepSmoke COMMAND microllm_model_s_train_smoke)
    set_tests_properties(ModelS.TrainStepSmoke PROPERTIES LABELS "cpu;model;slow;training")
endif()
