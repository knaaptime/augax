cmake_minimum_required(VERSION 3.18)
project(augax LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The compiler is whichever one CMake found on PATH.  On macOS its linker may still
# be older than the SDK: conda-forge's ld64 956.6 cannot read the stubs in SDK 27,
# which list the arm64e.x1 architecture, so it ignores libSystem and libc comes up
# undefined.  CMake's own compiler check misses this, because its test program calls
# nothing.  So link one that does, and if it fails, use an installed SDK that works.
# An SDK chosen through SDKROOT or CMAKE_OSX_SYSROOT is left alone.
if(APPLE)
    include(CheckCXXSourceCompiles)
    set(_link_probe "#include <cstdio>\nint main() { return std::puts(\"\") < 0; }")
    check_cxx_source_compiles("${_link_probe}" AUGAX_SDK_LINKS)
    if(NOT AUGAX_SDK_LINKS)
        set(_chosen_sdk "${CMAKE_OSX_SYSROOT}")
        set(_sdks "")
        if(NOT _chosen_sdk)
            execute_process(COMMAND xcrun --show-sdk-platform-path
                OUTPUT_VARIABLE _platform OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
            file(GLOB _sdks "${_platform}/Developer/SDKs/MacOSX*.sdk"
                "/Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk")
            list(SORT _sdks COMPARE NATURAL ORDER DESCENDING)
        endif()
        foreach(_sdk IN LISTS _sdks)
            set(CMAKE_OSX_SYSROOT "${_sdk}")
            unset(AUGAX_SDK_LINKS CACHE)
            check_cxx_source_compiles("${_link_probe}" AUGAX_SDK_LINKS)
            if(AUGAX_SDK_LINKS)
                message(STATUS "augax: the linker cannot read the default SDK; using ${_sdk}")
                break()
            endif()
        endforeach()
        if(NOT AUGAX_SDK_LINKS)
            string(REPLACE ";" ", " _tried "${_chosen_sdk}${_sdks}")
            message(FATAL_ERROR
                "${CMAKE_CXX_COMPILER} cannot link a C++ program against a macOS "
                "SDK (tried: ${_tried}). Its linker may be older than the SDK: "
                "install an older SDK with the Xcode Command Line Tools, or "
                "update the linker.")
        endif()
    endif()
endif()

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module
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

# The stable ABI lets one wheel serve every supported Python.
nanobind_add_module(_cpp NOMINSIZE STABLE_ABI src/cpp/augax.cpp)
target_include_directories(_cpp PRIVATE ${XLA_FFI_INCLUDE_DIR})

install(TARGETS _cpp LIBRARY DESTINATION augax)
