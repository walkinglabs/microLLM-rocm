add_library(microllm_autograd STATIC
    autograd.cpp
)
add_library(microllm::autograd ALIAS microllm_autograd)
set_target_properties(microllm_autograd PROPERTIES EXPORT_NAME autograd)

target_include_directories(microllm_autograd
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(microllm_autograd
    PUBLIC microllm::ops microllm::profiling microllm_project_options microllm_build_config
)

install(TARGETS microllm_autograd EXPORT microLLMTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
