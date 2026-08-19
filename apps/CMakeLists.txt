add_executable(microllm_train train.cpp)
target_link_libraries(microllm_train PRIVATE microllm::training microllm::io)

add_executable(microllm_runtime_info runtime_info.cpp)
target_link_libraries(microllm_runtime_info PRIVATE
    microllm::core microllm_build_config
)

add_executable(microllm_alignment alignment.cpp)
target_link_libraries(microllm_alignment PRIVATE
    microllm::model microllm::profiling microllm::runtime
)

install(TARGETS microllm_train microllm_runtime_info microllm_alignment
        RUNTIME DESTINATION bin)

if(MICROLLM_HAS_RCCL)
    add_executable(microllm_distributed_train distributed_train.cpp)
    target_link_libraries(microllm_distributed_train PRIVATE
        microllm::multi_gpu microllm::profiling
    )
    install(TARGETS microllm_distributed_train RUNTIME DESTINATION bin)
    if(BUILD_TESTING)
        add_test(NAME DistributedCLI.Smoke
                 COMMAND microllm_distributed_train --steps 1 --bucket-bytes 4096
                         --trace ${CMAKE_CURRENT_BINARY_DIR}/distributed-smoke.jsonl)
        set_tests_properties(DistributedCLI.Smoke PROPERTIES
                             LABELS "hip;rccl;multi_gpu;integration")
    endif()
endif()

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
