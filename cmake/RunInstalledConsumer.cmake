if(NOT DEFINED MICROLLM_BINARY_DIR OR
   NOT DEFINED MICROLLM_CONSUMER_SOURCE_DIR OR
   NOT DEFINED MICROLLM_CONSUMER_BINARY_DIR OR
   NOT DEFINED MICROLLM_INSTALL_PREFIX)
    message(FATAL_ERROR "installed consumer test arguments are incomplete")
endif()

set(MICROLLM_RELOCATED_PREFIX "${MICROLLM_INSTALL_PREFIX}-relocated")
file(REMOVE_RECURSE
    "${MICROLLM_INSTALL_PREFIX}"
    "${MICROLLM_RELOCATED_PREFIX}"
    "${MICROLLM_CONSUMER_BINARY_DIR}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${MICROLLM_BINARY_DIR}"
            --prefix "${MICROLLM_INSTALL_PREFIX}"
    RESULT_VARIABLE install_status
    OUTPUT_VARIABLE install_output
    ERROR_VARIABLE install_error)
if(NOT install_status EQUAL 0)
    message(FATAL_ERROR "microLLM install failed:\n${install_output}\n${install_error}")
endif()

# A package config must describe its current prefix, not remember the directory used
# during installation. Moving it before find_package catches absolute-path leaks.
file(RENAME "${MICROLLM_INSTALL_PREFIX}" "${MICROLLM_RELOCATED_PREFIX}")

execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${MICROLLM_CONSUMER_SOURCE_DIR}"
            -B "${MICROLLM_CONSUMER_BINARY_DIR}"
            "-DCMAKE_PREFIX_PATH=${MICROLLM_RELOCATED_PREFIX}"
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

# Required components are an API contract. A typo or an unavailable optional backend
# must fail at configure time rather than become a link error later.
set(missing_component_build "${MICROLLM_CONSUMER_BINARY_DIR}-missing-component")
file(REMOVE_RECURSE "${missing_component_build}")
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${MICROLLM_CONSUMER_SOURCE_DIR}/../missing_component"
            -B "${missing_component_build}"
            "-DCMAKE_PREFIX_PATH=${MICROLLM_RELOCATED_PREFIX}"
    RESULT_VARIABLE missing_component_status
    OUTPUT_VARIABLE missing_component_output
    ERROR_VARIABLE missing_component_error)
if(missing_component_status EQUAL 0)
    message(FATAL_ERROR "an unavailable required component was accepted")
endif()
if(NOT missing_component_output MATCHES "microLLM_FOUND.*FALSE" AND
   NOT missing_component_error MATCHES "microLLM_FOUND.*FALSE")
    message(FATAL_ERROR
        "missing-component failure did not come from package component validation:\n"
        "${missing_component_output}\n${missing_component_error}")
endif()

# A pre-1.0 package only promises compatibility inside its current minor line.
set(version_mismatch_build "${MICROLLM_CONSUMER_BINARY_DIR}-version-mismatch")
file(REMOVE_RECURSE "${version_mismatch_build}")
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${MICROLLM_CONSUMER_SOURCE_DIR}/../version_mismatch"
            -B "${version_mismatch_build}"
            "-DCMAKE_PREFIX_PATH=${MICROLLM_RELOCATED_PREFIX}"
    RESULT_VARIABLE version_mismatch_status
    OUTPUT_VARIABLE version_mismatch_output
    ERROR_VARIABLE version_mismatch_error)
if(version_mismatch_status EQUAL 0)
    message(FATAL_ERROR "an incompatible pre-1.0 minor version was accepted")
endif()
if(NOT version_mismatch_output MATCHES "compatible with requested version" AND
   NOT version_mismatch_error MATCHES "compatible with requested version")
    message(FATAL_ERROR
        "version-mismatch failure did not come from package version validation:\n"
        "${version_mismatch_output}\n${version_mismatch_error}")
endif()
