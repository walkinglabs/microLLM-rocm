# src/multi_gpu — RCCL 通信, 分布式训练
# TODO: N9 启用

if(rccl_FOUND)
    add_library(micro_llm_multi_gpu INTERFACE)
    target_include_directories(micro_llm_multi_gpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(micro_llm_multi_gpu INTERFACE rccl::rccl)
endif()
