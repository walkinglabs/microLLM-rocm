cmake_minimum_required(VERSION 3.25)
project(microLLMPackageConsumer LANGUAGES C CXX)

set(microllm_required_components inference model)
set(microllm_optional_components capi multi_gpu hip hipblaslt rccl rocwmma)
if(MICROLLM_REQUIRED_FEATURE)
    list(APPEND microllm_required_components "${MICROLLM_REQUIRED_FEATURE}")
    list(REMOVE_ITEM microllm_optional_components
        "${MICROLLM_REQUIRED_FEATURE}")
endif()

find_package(microLLM 0.1 CONFIG REQUIRED
    COMPONENTS ${microllm_required_components}
    OPTIONAL_COMPONENTS ${microllm_optional_components})

if(NOT DEFINED microLLM_VERSION OR microLLM_VERSION VERSION_LESS 0.1)
    message(FATAL_ERROR "package Config is missing valid version metadata")
endif()
if(NOT microLLM_VERSION_MAJOR EQUAL 0 OR
   NOT microLLM_VERSION_MINOR EQUAL 1 OR
   NOT microLLM_VERSION_PATCH EQUAL 0)
    message(FATAL_ERROR "package Config has inconsistent semantic-version fields")
endif()
if(NOT microLLM_CXX_STANDARD EQUAL 20)
    message(FATAL_ERROR "package Config reports the wrong C++ language standard")
endif()
if(NOT DEFINED microLLM_HIP_ARCHITECTURES)
    message(FATAL_ERROR "package Config is missing HIP architecture metadata")
endif()
set(microllm_unique_hip_architectures ${microLLM_HIP_ARCHITECTURES})
list(LENGTH microllm_unique_hip_architectures microllm_hip_architecture_count)
list(REMOVE_DUPLICATES microllm_unique_hip_architectures)
list(LENGTH microllm_unique_hip_architectures
    microllm_unique_hip_architecture_count)
if(NOT microllm_hip_architecture_count EQUAL
   microllm_unique_hip_architecture_count)
    message(FATAL_ERROR "package Config contains duplicate HIP architectures")
endif()
if(NOT DEFINED microLLM_AVAILABLE_FEATURE_COMPONENTS)
    message(FATAL_ERROR "package Config is missing feature-component metadata")
endif()
if(MICROLLM_REQUIRED_FEATURE AND
   NOT microLLM_${MICROLLM_REQUIRED_FEATURE}_FOUND)
    message(FATAL_ERROR
        "package did not satisfy required ${MICROLLM_REQUIRED_FEATURE} feature")
endif()
if((microLLM_WITH_RCCL AND NOT microLLM_BACKEND STREQUAL "HIP_RCCL") OR
   (microLLM_WITH_HIP AND NOT microLLM_WITH_RCCL AND
    NOT microLLM_BACKEND STREQUAL "HIP") OR
   (NOT microLLM_WITH_HIP AND NOT microLLM_BACKEND STREQUAL "CPU"))
    message(FATAL_ERROR "package Config reports inconsistent backend metadata")
endif()
if(NOT microLLM_DEFAULT_TARGET STREQUAL "microLLM::microLLM" OR
   NOT TARGET microLLM::microLLM)
    message(FATAL_ERROR "package Config is missing its default C++ SDK target")
endif()
foreach(microllm_exported_target IN LISTS microLLM_TARGETS)
    if(NOT TARGET "${microllm_exported_target}")
        message(FATAL_ERROR
            "package target metadata names a missing target: ${microllm_exported_target}")
    endif()
endforeach()
list(LENGTH microLLM_TARGETS microllm_target_count)
list(LENGTH microLLM_AVAILABLE_COMPONENTS microllm_component_count)
math(EXPR microllm_expected_target_count "${microllm_component_count} + 1")
if(NOT microllm_target_count EQUAL microllm_expected_target_count)
    message(FATAL_ERROR "package target metadata contains duplicates or omissions")
endif()
if(NOT DEFINED microLLM_WITH_ROCWMMA OR
   (microLLM_WITH_ROCWMMA AND NOT microLLM_WITH_HIP))
    message(FATAL_ERROR "package Config has invalid rocWMMA feature metadata")
endif()

if(NOT "inference" IN_LIST microLLM_AVAILABLE_COMPONENTS OR
   NOT microLLM_inference_FOUND OR NOT microLLM_model_FOUND)
    message(FATAL_ERROR "installed package component metadata is inconsistent")
endif()
if((microLLM_WITH_CAPI AND NOT "capi" IN_LIST microLLM_AVAILABLE_COMPONENTS) OR
   (NOT microLLM_WITH_CAPI AND "capi" IN_LIST microLLM_AVAILABLE_COMPONENTS))
    message(FATAL_ERROR "installed package C API metadata is inconsistent")
endif()

foreach(component runtime core profiling ops autograd io model training inference)
    if(NOT TARGET microLLM::${component})
        message(FATAL_ERROR "installed package is missing microLLM::${component}")
    endif()
endforeach()

# Repository-only warning and instrumentation compile flags must not leak through an
# installed or build-tree SDK. Instrumented static builds do carry their one required
# final-link runtime option.
get_target_property(microllm_consumer_compile_options
    microLLM::project_options INTERFACE_COMPILE_OPTIONS)
get_target_property(microllm_consumer_link_options
    microLLM::project_options INTERFACE_LINK_OPTIONS)
if(microllm_consumer_compile_options)
    message(FATAL_ERROR
        "microLLM package leaked repository-only compile options")
endif()
if(microLLM_WITH_SANITIZERS)
    if(NOT microllm_consumer_link_options STREQUAL "-fsanitize=address,undefined")
        message(FATAL_ERROR "sanitized package lost or widened its link requirement")
    endif()
elseif(microLLM_WITH_COVERAGE)
    if(NOT microllm_consumer_link_options STREQUAL "--coverage")
        message(FATAL_ERROR "coverage package lost or widened its link requirement")
    endif()
elseif(microllm_consumer_link_options)
    message(FATAL_ERROR "ordinary microLLM package leaked repository-only link options")
endif()
foreach(optional_component capi multi_gpu)
    if("${optional_component}" IN_LIST microLLM_AVAILABLE_COMPONENTS)
        if(NOT microLLM_${optional_component}_FOUND OR
           NOT TARGET microLLM::${optional_component})
            message(FATAL_ERROR
                "package metadata advertises a missing ${optional_component} target")
        endif()
    elseif(microLLM_${optional_component}_FOUND OR
           TARGET microLLM::${optional_component})
        message(FATAL_ERROR
            "package exposes unavailable ${optional_component} metadata or target")
    endif()
endforeach()

foreach(feature hip hipblaslt rccl rocwmma)
    string(TOUPPER "${feature}" feature_upper)
    if(feature STREQUAL "hipblaslt")
        set(feature_flag "microLLM_WITH_HIPBLASLT")
    elseif(feature STREQUAL "rocwmma")
        set(feature_flag "microLLM_WITH_ROCWMMA")
    else()
        set(feature_flag "microLLM_WITH_${feature_upper}")
    endif()
    if(${feature_flag})
        if(NOT feature IN_LIST microLLM_AVAILABLE_FEATURE_COMPONENTS OR
           NOT microLLM_${feature}_FOUND)
            message(FATAL_ERROR
                "package metadata omitted enabled ${feature} feature component")
        endif()
    elseif(feature IN_LIST microLLM_AVAILABLE_FEATURE_COMPONENTS OR
           microLLM_${feature}_FOUND)
        message(FATAL_ERROR
            "package metadata advertises disabled ${feature} feature component")
    endif()
endforeach()

add_executable(microllm_package_consumer main.cpp)
target_link_libraries(microllm_package_consumer PRIVATE microLLM::microLLM)
target_compile_features(microllm_package_consumer PRIVATE cxx_std_20)

if(microLLM_capi_FOUND)
    add_executable(microllm_c_package_consumer capi_main.c)
    target_link_libraries(microllm_c_package_consumer PRIVATE microLLM::capi)
endif()
