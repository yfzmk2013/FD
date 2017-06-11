cmake_minimum_required(VERSION 2.8.0)

project(FD)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
message(STATUS "C++11 support has been enabled by default.")
add_executable(FD_Test common.h main.cpp te.h)



