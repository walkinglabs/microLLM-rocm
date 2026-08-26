if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME PythonAPI.Basic
            COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${PROJECT_SOURCE_DIR}/python"
                "MICROLLM_LIBRARY=$<TARGET_FILE:microllm_c>"
                ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/python/tests/test_capi.py -v
        )
        set_tests_properties(PythonAPI.Basic PROPERTIES LABELS "cpu;python;integration")
        add_test(
            NAME PythonAPI.Profiling
            COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${PROJECT_SOURCE_DIR}/python"
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/python/tests/test_profiling_api.py -v
        )
        set_tests_properties(PythonAPI.Profiling PROPERTIES
            LABELS "cpu;python;profiling;unit")
    endif()
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/python/microllm
        DESTINATION lib/python)
