add_library(microllm_c SHARED microllm_c.cpp)
add_library(microllm::capi ALIAS microllm_c)
set_target_properties(microllm_c PROPERTIES
    OUTPUT_NAME microllm
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_compile_definitions(microllm_c PRIVATE MICROLLM_CAPI_EXPORTS=1)
target_include_directories(microllm_c
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(microllm_c PRIVATE microllm::ops)

install(TARGETS microllm_c LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_executable(microllm_capi_test capi_test.c)
    target_link_libraries(microllm_capi_test PRIVATE microllm_c)
    add_test(NAME CAPI.Basic COMMAND microllm_capi_test)
    set_tests_properties(CAPI.Basic PROPERTIES LABELS "cpu;capi;integration")
endif()
