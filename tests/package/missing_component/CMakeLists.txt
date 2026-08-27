cmake_minimum_required(VERSION 3.25)
project(microLLMMissingComponent LANGUAGES CXX)

set(MICROLLM_REQUESTED_COMPONENT "component_that_does_not_exist"
    CACHE STRING "Component intentionally requested by the package negative test")
find_package(microLLM 0.1 CONFIG REQUIRED
    COMPONENTS "${MICROLLM_REQUESTED_COMPONENT}")
