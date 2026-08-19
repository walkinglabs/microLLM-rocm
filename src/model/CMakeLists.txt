add_library(microllm_model STATIC
    config.cpp
    model.cpp
)
add_library(microllm::model ALIAS microllm_model)

target_include_directories(microllm_model
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(microllm_model
    PUBLIC microllm::autograd microllm_project_options microllm_build_config
)

install(TARGETS microllm_model ARCHIVE DESTINATION lib)
