if(NOT DEFINED MICROLLM_BINARY_DIR OR
   NOT DEFINED MICROLLM_EXAMPLE_SOURCE_DIR OR
   NOT DEFINED MICROLLM_EXAMPLE_BINARY_DIR OR
   NOT DEFINED MICROLLM_EXAMPLE_INSTALL_PREFIX)
    message(FATAL_ERROR "public package example test arguments are incomplete")
endif()

file(REMOVE_RECURSE
    "${MICROLLM_EXAMPLE_BINARY_DIR}"
    "${MICROLLM_EXAMPLE_INSTALL_PREFIX}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${MICROLLM_BINARY_DIR}"
            --prefix "${MICROLLM_EXAMPLE_INSTALL_PREFIX}"
    RESULT_VARIABLE install_status
    OUTPUT_VARIABLE install_output
    ERROR_VARIABLE install_error)
if(NOT install_status EQUAL 0)
    message(FATAL_ERROR
        "public example SDK install failed:\n${install_output}\n${install_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${MICROLLM_EXAMPLE_SOURCE_DIR}"
            -B "${MICROLLM_EXAMPLE_BINARY_DIR}"
            "-DCMAKE_PREFIX_PATH=${MICROLLM_EXAMPLE_INSTALL_PREFIX}"
    RESULT_VARIABLE configure_status
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_error)
if(NOT configure_status EQUAL 0)
    message(FATAL_ERROR
        "public example configure failed:\n${configure_output}\n${configure_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${MICROLLM_EXAMPLE_BINARY_DIR}"
    RESULT_VARIABLE build_status
    OUTPUT_VARIABLE build_output
    ERROR_VARIABLE build_error)
if(NOT build_status EQUAL 0)
    message(FATAL_ERROR
        "public example build failed:\n${build_output}\n${build_error}")
endif()

execute_process(
    COMMAND "${MICROLLM_EXAMPLE_BINARY_DIR}/microllm_package_example"
    RESULT_VARIABLE run_status
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error)
if(NOT run_status EQUAL 0 OR
   NOT run_output MATCHES "microLLM package example:")
    message(FATAL_ERROR
        "public example run failed:\n${run_output}\n${run_error}")
endif()
