cmake_minimum_required(VERSION 3.25)

project(microLLMCorePackageConsumer LANGUAGES CXX)

# Request and link only the lowest reusable Tensor layer. This project must not need
# the umbrella training/inference target to make the component package usable.
find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS core)

if(NOT microLLM_core_FOUND OR NOT TARGET microLLM::core)
    message(FATAL_ERROR "the package did not expose its requested core component")
endif()

add_executable(microllm_core_package_consumer main.cpp)
target_link_libraries(microllm_core_package_consumer PRIVATE microLLM::core)
