cmake_minimum_required(VERSION 3.25)

project(microLLMPackageExample LANGUAGES CXX)

# microLLM is a Config package. The caller selects a capability and links the
# corresponding imported target; include paths and lower-level libraries follow it.
find_package(microLLM 0.1 CONFIG REQUIRED COMPONENTS inference)

add_executable(microllm_package_example main.cpp)
target_link_libraries(microllm_package_example PRIVATE microLLM::inference)
