# src/autograd — 自动求导 (计算图, 反向传播)
# TODO: N5 启用

add_library(micro_llm_autograd INTERFACE)
target_include_directories(micro_llm_autograd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})