add_library(microllm_core STATIC
    device.cpp
    storage.cpp
    tensor.cpp
)
add_library(microllm::core ALIAS microllm_core)
set_target_properties(microllm_core PROPERTIES EXPORT_NAME core)

target_include_directories(microllm_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(microllm_core
    PUBLIC microllm_project_options microllm_build_config microllm::runtime
)

install(TARGETS microllm_core EXPORT microLLMTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
