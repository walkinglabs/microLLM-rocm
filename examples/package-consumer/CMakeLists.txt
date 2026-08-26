cmake_minimum_required(VERSION 3.25)

project(microLLMPackageExample LANGUAGES CXX)

# The default target exposes the complete single-device C++ SDK. Applications that
# need a smaller dependency surface can request a component and link its target.
find_package(microLLM 0.1 CONFIG REQUIRED)
message(STATUS "Using microLLM ${microLLM_VERSION} backend=${microLLM_BACKEND}")

add_executable(microllm_package_example main.cpp)
target_link_libraries(microllm_package_example PRIVATE microLLM::microLLM)
