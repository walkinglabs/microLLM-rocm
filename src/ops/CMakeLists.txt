add_library(microllm_ops STATIC
    ops.cpp
    low_level.cpp
    optimized.cpp
    tuning.cpp
    adamw_tuning.cpp
)
add_library(microllm::ops ALIAS microllm_ops)
set_target_properties(microllm_ops PROPERTIES EXPORT_NAME ops)

target_include_directories(microllm_ops
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(microllm_ops
    PUBLIC microllm::core microllm_project_options microllm_build_config
)
if(MICROLLM_HAS_HIP)
    target_sources(microllm_ops PRIVATE hip/basic_kernels.hip)
endif()
if(MICROLLM_HAS_HIPBLASLT)
    target_link_libraries(microllm_ops PRIVATE hip::host roc::hipblaslt)
endif()

install(TARGETS microllm_ops EXPORT microLLMTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
