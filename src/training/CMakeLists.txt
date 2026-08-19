add_library(microllm_training STATIC
    optimizer.cpp
    checkpoint.cpp
)
add_library(microllm::training ALIAS microllm_training)

target_include_directories(microllm_training
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(microllm_training
    PUBLIC microllm::autograd microllm_project_options microllm_build_config
)

install(TARGETS microllm_training ARCHIVE DESTINATION lib)
