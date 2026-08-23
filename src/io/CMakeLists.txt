add_library(microllm_io STATIC
    byte_tokenizer.cpp
    token_dataset.cpp
    bpe_tokenizer.cpp
    huggingface_bpe_tokenizer.cpp
    chat_template.cpp
    sft.cpp
    safetensors.cpp
)
add_library(microllm::io ALIAS microllm_io)
set_target_properties(microllm_io PROPERTIES EXPORT_NAME io)

target_include_directories(microllm_io
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(microllm_io
    PUBLIC microllm::core microllm_project_options microllm_build_config
)

install(TARGETS microllm_io EXPORT microLLMTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
