add_executable(microllm_train train.cpp)
target_link_libraries(microllm_train PRIVATE microllm::training microllm::io)

add_executable(microllm_runtime_info runtime_info.cpp)
target_link_libraries(microllm_runtime_info PRIVATE
    microllm::core microllm_build_config
)

install(TARGETS microllm_train microllm_runtime_info RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    set(MICROLLM_CLI_CHECKPOINT "${CMAKE_CURRENT_BINARY_DIR}/cli-smoke.ckpt")
    add_test(NAME TrainCLI.Save
             COMMAND microllm_train --data ${PROJECT_SOURCE_DIR}/data/generated-cycle.txt
                     --model tiny --device cpu --steps 2 --batch 1 --context 8
                     --seed 5 --learning-rate 0.001 --checkpoint ${MICROLLM_CLI_CHECKPOINT})
    set_tests_properties(TrainCLI.Save PROPERTIES
                         LABELS "cpu;training;integration" FIXTURES_SETUP cli_checkpoint)
    add_test(NAME TrainCLI.Resume
             COMMAND microllm_train --data ${PROJECT_SOURCE_DIR}/data/generated-cycle.txt
                     --model tiny --device cpu --steps 1 --batch 1 --context 8
                     --seed 5 --learning-rate 0.001 --resume ${MICROLLM_CLI_CHECKPOINT})
    set_tests_properties(TrainCLI.Resume PROPERTIES
                         LABELS "cpu;training;integration" FIXTURES_REQUIRED cli_checkpoint)
endif()
