file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

message(STATUS "Collected source files in dngawbtool:")
foreach(f ${srcs})
    message(STATUS "  ${f}")
endforeach()

add_library(layer SHARED ${srcs})

target_include_directories(layer
    PUBLIC
    ${PROJECT_SOURCE_DIR}/Layer
)

target_link_libraries(layer PUBLIC OpenMP::OpenMP_CXX)
