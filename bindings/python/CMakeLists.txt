if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME PythonAPI.Basic
            COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${PROJECT_SOURCE_DIR}/python"
                "MICROLLM_LIBRARY=$<TARGET_FILE:microllm_c>"
                ${Python3_EXECUTABLE} -m unittest discover
                -s ${PROJECT_SOURCE_DIR}/python/tests -v
        )
        set_tests_properties(PythonAPI.Basic PROPERTIES LABELS "cpu;python;integration")
    endif()
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/python/microllm
        DESTINATION lib/python)
