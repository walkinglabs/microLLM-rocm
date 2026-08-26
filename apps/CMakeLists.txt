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

add_executable(microllm_hf_inspect hf_inspect.cpp)
target_link_libraries(microllm_hf_inspect PRIVATE microllm::model)

add_executable(microllm_hf_infer hf_infer.cpp)
target_link_libraries(microllm_hf_infer PRIVATE
    microllm::model microllm::runtime microllm::inference microllm::profiling)

add_executable(microllm_hf_tokenize hf_tokenize.cpp)
target_link_libraries(microllm_hf_tokenize PRIVATE microllm::io)

add_executable(microllm_hf_train_step hf_train_step.cpp)
target_link_libraries(microllm_hf_train_step PRIVATE
    microllm::training microllm::model microllm::runtime)

add_executable(microllm_compare_safetensors compare_safetensors.cpp)
target_link_libraries(microllm_compare_safetensors PRIVATE microllm::io)

install(TARGETS microllm_train microllm_runtime_info microllm_alignment microllm_hf_inspect
        microllm_hf_infer
        microllm_hf_tokenize
        microllm_hf_train_step
        microllm_compare_safetensors
        RUNTIME DESTINATION bin)

if(MICROLLM_HAS_RCCL)
    add_executable(microllm_distributed_train distributed_train.cpp)
    target_link_libraries(microllm_distributed_train PRIVATE
        microllm::multi_gpu microllm::profiling
    )
    install(TARGETS microllm_distributed_train RUNTIME DESTINATION bin)
    add_executable(microllm_distributed_rank distributed_rank.cpp)
    target_link_libraries(microllm_distributed_rank PRIVATE
        microllm::multi_gpu microllm::training microllm::model microllm::runtime)
    install(TARGETS microllm_distributed_rank RUNTIME DESTINATION bin)
    if(BUILD_TESTING)
        add_test(NAME DistributedCLI.Smoke
                 COMMAND microllm_distributed_train --steps 1 --bucket-bytes 4096
                         --trace ${CMAKE_CURRENT_BINARY_DIR}/distributed-smoke.jsonl)
        set_tests_properties(DistributedCLI.Smoke PROPERTIES
                             LABELS "hip;rccl;multi_gpu;integration")
        add_test(NAME DistributedCLI.ModelSSmoke
                 COMMAND microllm_distributed_train --model model-s
                         --steps 1 --context 4 --batch 1
                         --bucket-bytes 4194304
                         --parameter-check-interval 1)
        set_tests_properties(DistributedCLI.ModelSSmoke PROPERTIES
                             LABELS "hip;rccl;multi_gpu;integration;model;slow")
    endif()
endif()

if(BUILD_TESTING)
    add_test(NAME HuggingFace.Qwen25Config
             COMMAND microllm_hf_inspect --config
                     ${PROJECT_SOURCE_DIR}/tests/fixtures/qwen25-0.5b-config.json)
    set_tests_properties(HuggingFace.Qwen25Config PROPERTIES
                         LABELS "cpu;huggingface;qwen;integration")
    add_test(NAME HuggingFace.Qwen3Config
             COMMAND microllm_hf_inspect --config
                     ${PROJECT_SOURCE_DIR}/tests/fixtures/qwen3-0.6b-config.json)
    set_tests_properties(HuggingFace.Qwen3Config PROPERTIES
                         LABELS "cpu;huggingface;qwen;qwen3;integration")
    add_test(NAME HuggingFace.DeepSeekDistillQwenConfig
             COMMAND microllm_hf_inspect --config
                     ${PROJECT_SOURCE_DIR}/tests/fixtures/deepseek-r1-distill-qwen-1.5b-config.json)
    set_tests_properties(HuggingFace.DeepSeekDistillQwenConfig PROPERTIES
                         LABELS "cpu;huggingface;deepseek;qwen;integration")
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
