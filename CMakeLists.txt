cmake_minimum_required(VERSION 3.5)
project(maxelem_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMAKE_BUILD_TYPE not set; setting to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build type is: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS "-march=native -Wall -Wextra -funroll-loops")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g")
add_executable(test.out src/test.cpp)
