add_library(microllm_c SHARED microllm_c.cpp)
add_library(microllm::capi ALIAS microllm_c)
add_library(microLLM::capi ALIAS microllm_c)
set_target_properties(microllm_c PROPERTIES
    EXPORT_NAME capi
    OUTPUT_NAME microllm
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_compile_definitions(microllm_c PRIVATE MICROLLM_CAPI_EXPORTS=1)
target_include_directories(microllm_c
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(microllm_c PRIVATE microllm::ops)

install(TARGETS microllm_c
    EXPORT microLLMTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(BUILD_TESTING)
    add_executable(microllm_capi_test capi_test.c)
    target_link_libraries(microllm_capi_test PRIVATE microllm_c)
    add_test(NAME CAPI.Basic COMMAND microllm_capi_test)
    set_tests_properties(CAPI.Basic PROPERTIES LABELS "cpu;capi;integration")
endif()
