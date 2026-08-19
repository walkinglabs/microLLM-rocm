add_library(microllm_multi_gpu STATIC communicator.cpp)
add_library(microllm::multi_gpu ALIAS microllm_multi_gpu)

target_include_directories(microllm_multi_gpu
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(microllm_multi_gpu
    PUBLIC microllm::ops microllm::runtime microllm_project_options microllm_build_config
    PRIVATE hip::host roc::rccl
)

install(TARGETS microllm_multi_gpu ARCHIVE DESTINATION lib)
