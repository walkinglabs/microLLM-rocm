if(NOT DEFINED MICROLLM_BINARY_DIR OR
   NOT DEFINED MICROLLM_CONSUMER_SOURCE_DIR OR
   NOT DEFINED MICROLLM_CONSUMER_BINARY_DIR OR
   NOT DEFINED MICROLLM_INSTALL_PREFIX)
    message(FATAL_ERROR "installed consumer test arguments are incomplete")
endif()

file(REMOVE_RECURSE "${MICROLLM_INSTALL_PREFIX}" "${MICROLLM_CONSUMER_BINARY_DIR}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${MICROLLM_BINARY_DIR}"
            --prefix "${MICROLLM_INSTALL_PREFIX}"
    RESULT_VARIABLE install_status
    OUTPUT_VARIABLE install_output
    ERROR_VARIABLE install_error)
if(NOT install_status EQUAL 0)
    message(FATAL_ERROR "microLLM install failed:\n${install_output}\n${install_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${MICROLLM_CONSUMER_SOURCE_DIR}"
            -B "${MICROLLM_CONSUMER_BINARY_DIR}"
            "-DCMAKE_PREFIX_PATH=${MICROLLM_INSTALL_PREFIX}"
    RESULT_VARIABLE configure_status
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_error)
if(NOT configure_status EQUAL 0)
    message(FATAL_ERROR "consumer configure failed:\n${configure_output}\n${configure_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${MICROLLM_CONSUMER_BINARY_DIR}"
    RESULT_VARIABLE build_status
    OUTPUT_VARIABLE build_output
    ERROR_VARIABLE build_error)
if(NOT build_status EQUAL 0)
    message(FATAL_ERROR "consumer build failed:\n${build_output}\n${build_error}")
endif()

execute_process(
    COMMAND "${MICROLLM_CONSUMER_BINARY_DIR}/microllm_package_consumer"
    RESULT_VARIABLE run_status
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error)
if(NOT run_status EQUAL 0 OR NOT run_output MATCHES "microLLM package consumer: pass")
    message(FATAL_ERROR "consumer run failed:\n${run_output}\n${run_error}")
endif()
