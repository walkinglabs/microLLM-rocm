add_library(microllm_training STATIC
    optimizer.cpp
    checkpoint.cpp
    trainer.cpp
)
add_library(microllm::training ALIAS microllm_training)
set_target_properties(microllm_training PROPERTIES EXPORT_NAME training)

target_include_directories(microllm_training
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(microllm_training
    PUBLIC microllm::autograd microllm::model microllm::io
           microllm_project_options microllm_build_config
)

install(TARGETS microllm_training EXPORT microLLMTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
