cmake_minimum_required(VERSION 3.16)
project(LULESH LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

add_executable(lulesh ${LULESH_SOURCES})
target_include_directories(lulesh PRIVATE ${CMAKE_SOURCE_DIR} ${GENERATED_DIR})
target_link_libraries(lulesh PUBLIC Kokkos::kokkos MPI::MPI_CXX)
target_compile_definitions(lulesh PRIVATE USE_MPI=1)
target_compile_options(lulesh PRIVATE)
