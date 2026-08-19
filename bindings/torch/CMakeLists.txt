find_package(Torch QUIET)

if(Torch_FOUND)
    add_library(microllm_torch_ops SHARED torch_ops.cpp)
    target_link_libraries(microllm_torch_ops PRIVATE microllm::ops "${TORCH_LIBRARIES}")
    target_include_directories(microllm_torch_ops PRIVATE ${TORCH_INCLUDE_DIRS})
    if(TORCH_CXX_FLAGS)
        separate_arguments(MICROLLM_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(microllm_torch_ops PRIVATE ${MICROLLM_TORCH_CXX_FLAGS})
    endif()
    set_target_properties(microllm_torch_ops PROPERTIES OUTPUT_NAME microllm_torch_ops)
    install(TARGETS microllm_torch_ops LIBRARY DESTINATION lib)

    if(BUILD_TESTING)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_executable(microllm_operator_oracle
                ${PROJECT_SOURCE_DIR}/tests/torch/operator_oracle.cpp
            )
            target_link_libraries(microllm_operator_oracle PRIVATE
                microllm::model microllm::training
            )

            add_test(
                NAME TorchOps.Basic
                COMMAND ${CMAKE_COMMAND} -E env
                    "PYTHONPATH=${PROJECT_SOURCE_DIR}/python"
                    "MICROLLM_TORCH_OP_LIBRARY=$<TARGET_FILE:microllm_torch_ops>"
                    ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/python/tests/test_torch_ops.py
            )
            set_tests_properties(TorchOps.Basic PROPERTIES LABELS "torch;integration")

            add_test(
                NAME TorchOps.OperatorParity
                COMMAND ${CMAKE_COMMAND} -E env
                    "MICROLLM_OPERATOR_ORACLE=$<TARGET_FILE:microllm_operator_oracle>"
                    ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/python/tests/test_operator_parity.py
            )
            set_tests_properties(TorchOps.OperatorParity PROPERTIES
                LABELS "torch;parity;graph;shape"
            )
        endif()
    endif()
    message(STATUS "PyTorch found: Custom Op target enabled")
elseif(MICROLLM_BUILD_TORCH_OPS_NORMALIZED STREQUAL "ON")
    message(FATAL_ERROR "PyTorch Custom Ops requested but TorchConfig.cmake was not found")
else()
    message(STATUS "PyTorch not found: Custom Op target not built")
endif()
