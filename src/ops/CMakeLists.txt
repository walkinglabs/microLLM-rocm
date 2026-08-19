add_library(microllm_ops STATIC
    ops.cpp
)
add_library(microllm::ops ALIAS microllm_ops)

target_include_directories(microllm_ops
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(microllm_ops
    PUBLIC microllm::core microllm_project_options microllm_build_config
)
if(MICROLLM_HAS_HIP)
    target_sources(microllm_ops PRIVATE hip/basic_kernels.hip)
endif()

install(TARGETS microllm_ops ARCHIVE DESTINATION lib)
