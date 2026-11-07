cmake_minimum_required(VERSION 3.18)
project(augax LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.SABIModule)

# XLA FFI headers ship with jaxlib (header-only, nothing to link).
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import jax.ffi; print(jax.ffi.include_dir())"
    OUTPUT_VARIABLE XLA_FFI_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)

# Locate nanobind via its Python package.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
find_package(nanobind REQUIRED CONFIG)

# The stable ABI needs Python 3.12; under 3.11 nanobind builds an ordinary module.
nanobind_add_module(_cpp NOMINSIZE STABLE_ABI src/cpp/augax.cpp)
target_include_directories(_cpp PRIVATE ${XLA_FFI_INCLUDE_DIR})

install(TARGETS _cpp LIBRARY DESTINATION augax)
