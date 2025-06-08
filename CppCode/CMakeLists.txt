cmake_minimum_required(VERSION 3.14)
project(knn_core_project LANGUAGES CXX)

# Use modern C++ and ensure compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and NumPy
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Add OpenMP (optional but highly recommended for performance)
find_package(OpenMP)

# Fetch pybind11 if not already available
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Create the module
pybind11_add_module(knn_core knn_core.cpp)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
  target_link_libraries(knn_core PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(knn_core PRIVATE _OPENMP)
endif()

# Link Python and NumPy
target_include_directories(knn_core PRIVATE
  ${Python3_INCLUDE_DIRS}
  ${Python3_NumPy_INCLUDE_DIRS}
)