add_library(microllm_inference STATIC
    generator.cpp
    scheduler.cpp
)
add_library(microllm::inference ALIAS microllm_inference)

target_include_directories(microllm_inference
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(microllm_inference
    PUBLIC microllm::model microllm_project_options microllm_build_config
)

install(TARGETS microllm_inference ARCHIVE DESTINATION lib)
