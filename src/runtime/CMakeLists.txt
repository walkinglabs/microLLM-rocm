add_library(microllm_runtime STATIC
    runtime.cpp
)
add_library(microllm::runtime ALIAS microllm_runtime)

target_include_directories(microllm_runtime
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(microllm_runtime
    PUBLIC microllm_project_options microllm_build_config
)
if(MICROLLM_HAS_HIP)
    target_link_libraries(microllm_runtime PRIVATE hip::host)
endif()

install(TARGETS microllm_runtime ARCHIVE DESTINATION lib)
