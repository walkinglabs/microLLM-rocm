add_library(microllm_profiling STATIC
    trace.cpp
)
add_library(microllm::profiling ALIAS microllm_profiling)
set_target_properties(microllm_profiling PROPERTIES EXPORT_NAME profiling)

target_include_directories(microllm_profiling
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(microllm_profiling
    PUBLIC microllm::core microllm::runtime microllm_project_options microllm_build_config
)

install(TARGETS microllm_profiling EXPORT microLLMTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
