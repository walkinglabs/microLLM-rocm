add_library(microllm_multi_gpu STATIC
    communicator.cpp
    gradient_bucket.cpp
    data_parallel.cpp
)
add_library(microllm::multi_gpu ALIAS microllm_multi_gpu)
set_target_properties(microllm_multi_gpu PROPERTIES EXPORT_NAME multi_gpu)

target_include_directories(microllm_multi_gpu
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(microllm_multi_gpu
    PUBLIC microllm::model microllm::training microllm::profiling
           microllm::ops microllm::runtime microllm_project_options microllm_build_config
    PRIVATE hip::host roc::rccl
)

install(TARGETS microllm_multi_gpu EXPORT microLLMTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
