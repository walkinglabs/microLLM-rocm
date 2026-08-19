# src/inference — KV Cache, 自回归生成
# TODO: N8 启用

add_library(micro_llm_inference INTERFACE)
target_include_directories(micro_llm_inference INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
