add_executable(microllm_tensor_ppm tensor_ppm.cpp)
target_link_libraries(microllm_tensor_ppm PRIVATE microllm::core)

add_executable(microllm_cpu_hip_compare cpu_hip_compare.cpp)
target_link_libraries(microllm_cpu_hip_compare PRIVATE microllm::ops microllm::runtime)

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

add_executable(microllm_tiny_hip_train tiny_hip_train.cpp)
target_link_libraries(microllm_tiny_hip_train PRIVATE microllm::training microllm::runtime)

add_executable(microllm_model_m_hip_train_smoke model_m_hip_train_smoke.cpp)
target_link_libraries(microllm_model_m_hip_train_smoke PRIVATE
    microllm::training microllm::runtime
)

add_executable(microllm_tiny_sft tiny_sft.cpp)
target_link_libraries(microllm_tiny_sft PRIVATE microllm::training microllm::io)

if(BUILD_TESTING)
    add_test(NAME Examples.TensorPpm COMMAND microllm_tensor_ppm
             "${CMAKE_CURRENT_BINARY_DIR}/tensor.ppm")
    set_tests_properties(Examples.TensorPpm PROPERTIES LABELS "cpu;example;smoke")
    if(MICROLLM_HAS_HIP)
        add_test(NAME Examples.CpuHipCompare COMMAND microllm_cpu_hip_compare)
        set_tests_properties(Examples.CpuHipCompare PROPERTIES LABELS "hip;example;smoke")
        add_test(NAME ModelS.HipForwardSmoke COMMAND microllm_model_s_hip_smoke)
        set_tests_properties(ModelS.HipForwardSmoke PROPERTIES LABELS "hip;model;slow")
        add_test(NAME Training.TinyHipSmoke COMMAND microllm_tiny_hip_train)
        set_tests_properties(Training.TinyHipSmoke PROPERTIES LABELS "hip;model;training;smoke")
        add_test(NAME ModelM.HipTrainStepSmoke COMMAND microllm_model_m_hip_train_smoke)
        set_tests_properties(ModelM.HipTrainStepSmoke PROPERTIES
                             LABELS "hip;model;training;slow")
    endif()
    add_test(NAME Training.TinySftSmoke COMMAND microllm_tiny_sft)
    set_tests_properties(Training.TinySftSmoke PROPERTIES LABELS "cpu;model;sft;smoke")
    add_test(NAME Training.TinyOverfitSmoke COMMAND microllm_tiny_overfit)
    set_tests_properties(Training.TinyOverfitSmoke PROPERTIES LABELS "cpu;model;smoke")
    add_test(NAME ModelS.ForwardSmoke COMMAND microllm_model_s_smoke)
    set_tests_properties(ModelS.ForwardSmoke PROPERTIES LABELS "cpu;model;slow")
    add_test(NAME ModelS.TrainStepSmoke COMMAND microllm_model_s_train_smoke)
    set_tests_properties(ModelS.TrainStepSmoke PROPERTIES LABELS "cpu;model;slow;training")
endif()
