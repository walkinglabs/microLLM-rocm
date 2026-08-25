cmake_minimum_required(VERSION 3.25)
project(microLLMCOnlyPackageConsumer LANGUAGES C)

# This is intentionally a C-only project. It proves that consumers of the stable
# ABI do not have to enable C++ or understand the C++ implementation libraries.
find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS capi)

if(CMAKE_CXX_COMPILER_LOADED)
    message(FATAL_ERROR "the C-only package consumer unexpectedly enabled C++")
endif()
if(NOT microLLM_capi_FOUND OR NOT TARGET microLLM::capi)
    message(FATAL_ERROR "the package did not expose its requested C API component")
endif()

add_executable(microllm_c_only_package_consumer main.c)
target_link_libraries(microllm_c_only_package_consumer PRIVATE microLLM::capi)
target_compile_features(microllm_c_only_package_consumer PRIVATE c_std_11)
