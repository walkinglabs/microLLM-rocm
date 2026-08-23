add_library(microllm_inference STATIC
    generator.cpp
    kv_cache.cpp
    scheduler.cpp
)
add_library(microllm::inference ALIAS microllm_inference)
set_target_properties(microllm_inference PROPERTIES EXPORT_NAME inference)

target_include_directories(microllm_inference
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(microllm_inference
    PUBLIC microllm::model microllm_project_options microllm_build_config
)

install(TARGETS microllm_inference EXPORT microLLMTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
