cmake_minimum_required(VERSION 3.25)
project(microLLMPackageConsumer LANGUAGES CXX)

find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS inference model)

if(NOT "inference" IN_LIST microLLM_AVAILABLE_COMPONENTS OR
   NOT microLLM_inference_FOUND OR NOT microLLM_model_FOUND)
    message(FATAL_ERROR "installed package component metadata is inconsistent")
endif()

foreach(component runtime core profiling ops autograd io model training inference)
    if(NOT TARGET microLLM::${component})
        message(FATAL_ERROR "installed package is missing microLLM::${component}")
    endif()
endforeach()
if("multi_gpu" IN_LIST microLLM_AVAILABLE_COMPONENTS AND
   NOT TARGET microLLM::multi_gpu)
    message(FATAL_ERROR "RCCL package metadata advertises a missing multi_gpu target")
endif()

add_executable(microllm_package_consumer main.cpp)
target_link_libraries(microllm_package_consumer PRIVATE microLLM::inference)
target_compile_features(microllm_package_consumer PRIVATE cxx_std_20)
