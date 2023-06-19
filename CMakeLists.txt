cmake_minimum_required(VERSION 3.14)
project(matrixd)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE SOURCES "tests/*.cpp")

set(CMAKE_CXX_FLAGS "-O3")

add_executable(tests ${SOURCES})

find_package(BLAS)

target_include_directories(tests PRIVATE "include" ${BLAS_INCLUDE_DIRS})
target_link_libraries(
    tests
    GTest::gtest_main
    ${BLAS_LIBRARIES}
)
