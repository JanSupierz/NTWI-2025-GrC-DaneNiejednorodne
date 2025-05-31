cmake_minimum_required(VERSION 3.14)
project(knn_core_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1  # or latest stable tag
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(knn_core knn_core.cpp)
