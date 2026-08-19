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

target_include_directories(microllm_io
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(microllm_io
    PUBLIC microllm::core microllm_project_options microllm_build_config
)

install(TARGETS microllm_io ARCHIVE DESTINATION lib)
