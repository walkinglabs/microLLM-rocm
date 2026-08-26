if(NOT DEFINED MICROLLM_SOURCE_DIR OR NOT DEFINED MICROLLM_BINARY_DIR)
    message(FATAL_ERROR "invalid-destination test arguments are incomplete")
endif()

set(invalid_build "${MICROLLM_BINARY_DIR}/package-invalid-destination")
file(REMOVE_RECURSE "${invalid_build}")
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${MICROLLM_SOURCE_DIR}"
            -B "${invalid_build}"
            -DMICROLLM_ENABLE_HIP=OFF
            -DMICROLLM_BUILD_TESTS=OFF
            -DMICROLLM_BUILD_APPS=OFF
            -DMICROLLM_BUILD_EXAMPLES=OFF
            -DMICROLLM_BUILD_CAPI=OFF
            -DMICROLLM_BUILD_PYTHON=OFF
            -DMICROLLM_BUILD_BENCHMARKS=OFF
            -DMICROLLM_BUILD_TORCH_OPS=OFF
            "-DMICROLLM_INSTALL_CMAKEDIR=${invalid_build}/outside-prefix"
    RESULT_VARIABLE configure_status
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_error)
if(configure_status EQUAL 0)
    message(FATAL_ERROR "an absolute CMake package destination was accepted")
endif()
if(NOT configure_output MATCHES "must be a non-empty path relative" AND
   NOT configure_error MATCHES "must be a non-empty path relative")
    message(FATAL_ERROR
        "the invalid destination failed for an unrelated reason:\n"
        "${configure_output}\n${configure_error}")
endif()
