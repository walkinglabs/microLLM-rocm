cmake_minimum_required(VERSION 3.25)

project(microLLM
    VERSION 0.1.0
    DESCRIPTION "A small, teachable LLM training and inference engine for AMD ROCm"
    LANGUAGES C CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MICROLLM_ENABLE_HIP "AUTO" CACHE STRING "Enable HIP support: AUTO, ON, or OFF")
set_property(CACHE MICROLLM_ENABLE_HIP PROPERTY STRINGS AUTO ON OFF)
set(MICROLLM_HIP_ARCHITECTURES "" CACHE STRING "Explicit CMake HIP architectures, e.g. gfx1100")
option(MICROLLM_ENABLE_RCCL "Enable RCCL distributed support" OFF)
option(MICROLLM_ENABLE_HIPBLASLT "Enable hipBLASLt optimized operators" ON)
option(MICROLLM_BUILD_TESTS "Build tests" ON)
option(MICROLLM_BUILD_EXAMPLES "Build runnable examples" ON)
option(MICROLLM_BUILD_CAPI "Build the stable C API shared library" ON)
option(MICROLLM_BUILD_PYTHON "Enable tests/install rules for the ctypes Python API" ON)
option(MICROLLM_BUILD_BENCHMARKS "Build reproducible micro and end-to-end benchmarks" ON)
set(MICROLLM_BUILD_TORCH_OPS "AUTO" CACHE STRING "Build PyTorch Custom Ops: AUTO, ON, or OFF")
set_property(CACHE MICROLLM_BUILD_TORCH_OPS PROPERTY STRINGS AUTO ON OFF)
option(MICROLLM_ENABLE_SANITIZERS "Enable ASan and UBSan for host code" OFF)
option(MICROLLM_ENABLE_COVERAGE "Enable GCC/Clang line and branch coverage" OFF)
set(MICROLLM_SAFETENSORS_PYTHON "" CACHE FILEPATH
    "Python interpreter with torch and official safetensors for interoperability tests")

string(TOUPPER "${MICROLLM_ENABLE_HIP}" MICROLLM_ENABLE_HIP_NORMALIZED)
if(NOT MICROLLM_ENABLE_HIP_NORMALIZED MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "MICROLLM_ENABLE_HIP must be AUTO, ON, or OFF")
endif()

set(MICROLLM_HAS_HIP OFF)
if(NOT MICROLLM_ENABLE_HIP_NORMALIZED STREQUAL "OFF")
    if(MICROLLM_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "${MICROLLM_HIP_ARCHITECTURES}" CACHE STRING "" FORCE)
    endif()
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        find_package(HIP REQUIRED)
        set(MICROLLM_HAS_HIP ON)
    elseif(MICROLLM_ENABLE_HIP_NORMALIZED STREQUAL "ON")
        message(FATAL_ERROR "HIP was requested but no HIP compiler was found")
    endif()
endif()

set(MICROLLM_HAS_HIPBLASLT OFF)
if(MICROLLM_HAS_HIP AND MICROLLM_ENABLE_HIPBLASLT)
    find_package(hipblaslt QUIET)
    if(hipblaslt_FOUND)
        set(MICROLLM_HAS_HIPBLASLT ON)
    endif()
endif()

set(MICROLLM_HAS_RCCL OFF)
if(MICROLLM_ENABLE_RCCL)
    if(NOT MICROLLM_HAS_HIP)
        message(FATAL_ERROR "RCCL requires HIP support")
    endif()
    find_package(rccl REQUIRED)
    set(MICROLLM_HAS_RCCL ON)
endif()

add_library(microllm_project_options INTERFACE)
target_compile_features(microllm_project_options INTERFACE cxx_std_20)
if(MICROLLM_ENABLE_SANITIZERS AND MICROLLM_ENABLE_COVERAGE)
    message(FATAL_ERROR "sanitizers and coverage must use separate builds")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # These are repository build policies, not SDK usage requirements. Directory
    # options compile every target in this checkout without leaking into either the
    # build-tree or installed CMake package.
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if(MICROLLM_ENABLE_SANITIZERS)
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
        # Instrumented static objects require the runtime at the final consumer link.
        # Propagate only that link requirement, never the repository compile flags.
        target_link_options(microllm_project_options INTERFACE
            -fsanitize=address,undefined)
    endif()
    if(MICROLLM_ENABLE_COVERAGE)
        add_compile_options(--coverage -O0 -g)
        add_link_options(--coverage)
        target_link_options(microllm_project_options INTERFACE --coverage)
    endif()
elseif(MICROLLM_ENABLE_COVERAGE)
    message(FATAL_ERROR "coverage currently requires GCC or Clang")
endif()

add_library(microllm_build_config INTERFACE)
target_compile_definitions(microllm_build_config INTERFACE
    MICROLLM_VERSION="${PROJECT_VERSION}"
    MICROLLM_HAS_HIP=$<BOOL:${MICROLLM_HAS_HIP}>
    MICROLLM_HAS_HIPBLASLT=$<BOOL:${MICROLLM_HAS_HIPBLASLT}>
    MICROLLM_HAS_RCCL=$<BOOL:${MICROLLM_HAS_RCCL}>
)
set_target_properties(microllm_project_options PROPERTIES EXPORT_NAME project_options)
set_target_properties(microllm_build_config PROPERTIES EXPORT_NAME build_config)

add_subdirectory(src/runtime)
add_subdirectory(src/core)
add_subdirectory(src/profiling)
add_subdirectory(src/ops)
add_subdirectory(src/autograd)
add_subdirectory(src/model)
add_subdirectory(src/io)
add_subdirectory(src/training)
add_subdirectory(src/inference)
if(MICROLLM_HAS_RCCL)
    add_subdirectory(src/multi_gpu)
endif()

if(MICROLLM_BUILD_TESTS)
    include(CTest)
endif()

add_subdirectory(apps)

if(MICROLLM_BUILD_CAPI)
    add_subdirectory(bindings/capi)
    if(MICROLLM_BUILD_PYTHON)
        add_subdirectory(bindings/python)
    endif()
endif()

string(TOUPPER "${MICROLLM_BUILD_TORCH_OPS}" MICROLLM_BUILD_TORCH_OPS_NORMALIZED)
if(NOT MICROLLM_BUILD_TORCH_OPS_NORMALIZED MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "MICROLLM_BUILD_TORCH_OPS must be AUTO, ON, or OFF")
endif()
if(NOT MICROLLM_BUILD_TORCH_OPS_NORMALIZED STREQUAL "OFF")
    add_subdirectory(bindings/torch)
endif()

if(MICROLLM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(MICROLLM_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(MICROLLM_BUILD_TESTS)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    add_subdirectory(tests)
endif()

install(DIRECTORY include/microllm DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS microllm_project_options microllm_build_config
        EXPORT microLLMTargets)

set(MICROLLM_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/microLLM"
    CACHE STRING "Install path for the microLLM CMake package")
set(MICROLLM_PACKAGE_COMPONENTS
    "runtime;core;profiling;ops;autograd;io;model;training;inference")
if(MICROLLM_BUILD_CAPI)
    list(APPEND MICROLLM_PACKAGE_COMPONENTS capi)
endif()
if(MICROLLM_HAS_RCCL)
    list(APPEND MICROLLM_PACKAGE_COMPONENTS multi_gpu)
endif()
configure_package_config_file(
    cmake/microLLMConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/microLLMConfig.cmake"
    INSTALL_DESTINATION "${MICROLLM_INSTALL_CMAKEDIR}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/microLLMConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMinorVersion
)
install(EXPORT microLLMTargets
        FILE microLLMTargets.cmake
        NAMESPACE microLLM::
        DESTINATION "${MICROLLM_INSTALL_CMAKEDIR}")
# Keep a developer-facing Config package beside the configured build. This lets a
# separate project consume an already-built checkout without copying sources or
# installing it first. The installed package below remains the deployment path.
export(EXPORT microLLMTargets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/microLLMTargets.cmake"
       NAMESPACE microLLM::)
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/microLLMConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/microLLMConfigVersion.cmake"
        DESTINATION "${MICROLLM_INSTALL_CMAKEDIR}")
install(FILES LICENSE README.md
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/microLLM")

message(STATUS "")
message(STATUS "microLLM-rocm ${PROJECT_VERSION}")
message(STATUS "  HIP:          ${MICROLLM_HAS_HIP}")
message(STATUS "  HIP arch:     ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "  hipBLASLt:    ${MICROLLM_HAS_HIPBLASLT}")
message(STATUS "  RCCL:         ${MICROLLM_HAS_RCCL}")
message(STATUS "  Tests:        ${MICROLLM_BUILD_TESTS}")
message(STATUS "  Examples:     ${MICROLLM_BUILD_EXAMPLES}")
message(STATUS "  C API:        ${MICROLLM_BUILD_CAPI}")
message(STATUS "  Sanitizers:   ${MICROLLM_ENABLE_SANITIZERS}")
message(STATUS "  Coverage:     ${MICROLLM_ENABLE_COVERAGE}")
message(STATUS "  CMake package: ${MICROLLM_INSTALL_CMAKEDIR}")
message(STATUS "")
