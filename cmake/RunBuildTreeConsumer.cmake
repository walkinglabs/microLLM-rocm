if(NOT DEFINED MICROLLM_BINARY_DIR OR
   NOT DEFINED MICROLLM_CONSUMER_SOURCE_DIR OR
   NOT DEFINED MICROLLM_CONSUMER_BINARY_DIR)
    message(FATAL_ERROR "build-tree consumer test arguments are incomplete")
endif()

file(REMOVE_RECURSE "${MICROLLM_CONSUMER_BINARY_DIR}")
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${MICROLLM_CONSUMER_SOURCE_DIR}"
            -B "${MICROLLM_CONSUMER_BINARY_DIR}"
            "-DmicroLLM_DIR=${MICROLLM_BINARY_DIR}"
    RESULT_VARIABLE configure_status
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_error)
if(NOT configure_status EQUAL 0)
    message(FATAL_ERROR
        "build-tree consumer configure failed:\n${configure_output}\n${configure_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${MICROLLM_CONSUMER_BINARY_DIR}"
    RESULT_VARIABLE build_status
    OUTPUT_VARIABLE build_output
    ERROR_VARIABLE build_error)
if(NOT build_status EQUAL 0)
    message(FATAL_ERROR
        "build-tree consumer build failed:\n${build_output}\n${build_error}")
endif()

execute_process(
    COMMAND "${MICROLLM_CONSUMER_BINARY_DIR}/microllm_package_consumer"
    RESULT_VARIABLE run_status
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error)
if(NOT run_status EQUAL 0 OR
   NOT run_output MATCHES "microLLM package consumer: pass")
    message(FATAL_ERROR
        "build-tree consumer run failed:\n${run_output}\n${run_error}")
endif()

set(c_consumer "${MICROLLM_CONSUMER_BINARY_DIR}/microllm_c_package_consumer")
if(EXISTS "${c_consumer}")
    execute_process(
        COMMAND "${c_consumer}"
        RESULT_VARIABLE c_run_status
        OUTPUT_VARIABLE c_run_output
        ERROR_VARIABLE c_run_error)
    if(NOT c_run_status EQUAL 0 OR
       NOT c_run_output MATCHES "microLLM C package consumer: pass")
        message(FATAL_ERROR
            "build-tree C consumer run failed:\n${c_run_output}\n${c_run_error}")
    endif()
endif()

set(missing_component_build "${MICROLLM_CONSUMER_BINARY_DIR}-missing-component")
file(REMOVE_RECURSE "${missing_component_build}")
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${MICROLLM_CONSUMER_SOURCE_DIR}/../missing_component"
            -B "${missing_component_build}"
            "-DmicroLLM_DIR=${MICROLLM_BINARY_DIR}"
    RESULT_VARIABLE missing_component_status
    OUTPUT_VARIABLE missing_component_output
    ERROR_VARIABLE missing_component_error)
if(missing_component_status EQUAL 0)
    message(FATAL_ERROR "build-tree package accepted an unavailable component")
endif()
if(NOT missing_component_output MATCHES "microLLM_FOUND.*FALSE" AND
   NOT missing_component_error MATCHES "microLLM_FOUND.*FALSE")
    message(FATAL_ERROR
        "build-tree missing-component failure bypassed component validation:\n"
        "${missing_component_output}\n${missing_component_error}")
endif()
if(NOT missing_component_output MATCHES "Available components:" AND
   NOT missing_component_error MATCHES "Available components:")
    message(FATAL_ERROR
        "build-tree missing-component failure omitted available components:\n"
        "${missing_component_output}\n${missing_component_error}")
endif()

set(version_mismatch_build "${MICROLLM_CONSUMER_BINARY_DIR}-version-mismatch")
file(REMOVE_RECURSE "${version_mismatch_build}")
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${MICROLLM_CONSUMER_SOURCE_DIR}/../version_mismatch"
            -B "${version_mismatch_build}"
            "-DmicroLLM_DIR=${MICROLLM_BINARY_DIR}"
    RESULT_VARIABLE version_mismatch_status
    OUTPUT_VARIABLE version_mismatch_output
    ERROR_VARIABLE version_mismatch_error)
if(version_mismatch_status EQUAL 0)
    message(FATAL_ERROR "build-tree package accepted an incompatible version")
endif()
if(NOT version_mismatch_output MATCHES "compatible with requested version" AND
   NOT version_mismatch_error MATCHES "compatible with requested version")
    message(FATAL_ERROR
        "build-tree version failure bypassed package version validation:\n"
        "${version_mismatch_output}\n${version_mismatch_error}")
endif()
