# src/ops — CPU 和 HIP 算子

# CPU 算子库
add_library(micro_llm_ops_cpu INTERFACE)
target_include_directories(micro_llm_ops_cpu INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu
)

# HIP 算子库 (仅当 HIP 可用时)
if(HIP_FOUND)
    add_library(micro_llm_ops_hip INTERFACE)
    target_include_directories(micro_llm_ops_hip INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/hip
    )
    if(hipblaslt_FOUND)
        target_link_libraries(micro_llm_ops_hip INTERFACE hipblaslt::hipblaslt)
    endif()
endif()