cmake_minimum_required(VERSION 3.16)
project(LULESH LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#include(${CMAKE_SOURCE_DIR}/config.cmake)

option(Charm_ENABLE_GPU "Build LULESH with Charm++/Kokkos GPU backend" ON)

set(Kokkos_ROOT ${KOKKOS_DIR})
find_package(Kokkos REQUIRED CONFIG)
find_package(MPI REQUIRED COMPONENTS CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LULESH_SOURCES
    lulesh-comm.cc
    lulesh-init.cc
    lulesh-util.cc
    lulesh-viz.cc
    lulesh.cc
)

set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)

#add_custom_command(
#    OUTPUT ${GENERATED_DIR}/miniMD.decl.h ${GENERATED_DIR}/miniMD.def.h
#    COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_DIR}
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/ljs.ci ${GENERATED_DIR}/ljs.ci
#    COMMAND ${CMAKE_COMMAND} -E chdir ${GENERATED_DIR} ${CHARMC} ljs.ci
#    DEPENDS ${CMAKE_SOURCE_DIR}/ljs.ci
#    COMMENT "Generating Charm++ interface for miniMD"
#    VERBATIM
#)

if(Charm_ENABLE_GPU)
    add_executable(lulesh ${LULESH_SOURCES})
    target_include_directories(lulesh PRIVATE ${CMAKE_SOURCE_DIR} ${GENERATED_DIR})
    target_link_libraries(lulesh PUBLIC Kokkos::kokkos MPI::MPI_CXX)
    string(REPLACE " " ";" GPU_COMPILE_OPTIONS "${GPU_OPTS}")
    target_compile_definitions(lulesh PRIVATE USE_MPI=1)
    target_compile_options(lulesh PRIVATE ${GPU_COMPILE_OPTIONS})
else()
    set(CMAKE_CXX_COMPILER "${CHARMC}")
    add_executable(miniMD ${MINIMD_SOURCES} ${GENERATED_DIR}/miniMD.decl.h ${GENERATED_DIR}/miniMD.def.h)
    target_include_directories(miniMD PRIVATE ${CMAKE_SOURCE_DIR} ${GENERATED_DIR} ${CHARM_DIR}/include)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CPU_OPTS} ${LD_OPTS}")
    target_link_libraries(miniMD PRIVATE Kokkos::kokkos)
endif()
