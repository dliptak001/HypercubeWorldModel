cmake_minimum_required(VERSION 3.21)
project(HypercubeWorldModelPython LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── pybind11 ──
find_package(pybind11 CONFIG REQUIRED)

# ── Optimization flags (match the main project / sibling python builds) ──
# HYPERCUBE_ARCH controls -march. Defaults to "native" for local dev builds.
# cibuildwheel overrides to "x86-64-v2" (x86_64) or "none" (ARM, MSVC).
set(HYPERCUBE_ARCH "native" CACHE STRING "Target architecture for -march (native, x86-64-v2, none)")

if(MSVC)
    add_compile_options(/O2 /fp:fast)
else()
    add_compile_options(-O3 -ffast-math)
    if(NOT HYPERCUBE_ARCH STREQUAL "none")
        if(HYPERCUBE_ARCH STREQUAL "native")
            add_compile_options(-march=native -mtune=native)
        else()
            add_compile_options(-march=${HYPERCUBE_ARCH} -mtune=generic)
        endif()
    endif()
    add_compile_options(-Wall -Wextra -Wno-unknown-pragmas)
endif()

# ── Core sources compiled directly into the module (PIC required) ──
# The SDK is WorldModel plus Decoder; both pull in LCN, and WorldModel
# pulls in Encoder and Predictor. ThreadPool is a test helper, not SDK.
set(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CORE_SOURCES
    ${CORE_DIR}/Encoder/Encoder.cpp
    ${CORE_DIR}/LCN/LCN.cpp
    ${CORE_DIR}/LCN/LCNTraining.cpp
    ${CORE_DIR}/Predictor/Predictor.cpp
    ${CORE_DIR}/Decoder/Decoder.cpp
    ${CORE_DIR}/WorldModel/WorldModel.cpp
)

# ── Package version (single source: hypercube_worldmodel/_version.py) ──
set(_HWM_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/hypercube_worldmodel/_version.py")
set(HYPERCUBE_WORLDMODEL_VERSION "")
file(STRINGS "${_HWM_VERSION_FILE}" _HWM_VERSION_LINES)
foreach(_line IN LISTS _HWM_VERSION_LINES)
    # Strip CR (Windows) and match: __version__ = "x.y.z"
    string(REPLACE "\r" "" _line "${_line}")
    if(_line MATCHES "^__version__[ \t]*=[ \t]*\"([^\"]+)\"")
        set(HYPERCUBE_WORLDMODEL_VERSION "${CMAKE_MATCH_1}")
        break()
    endif()
endforeach()
if(HYPERCUBE_WORLDMODEL_VERSION STREQUAL "")
    message(FATAL_ERROR
        "Could not parse __version__ from hypercube_worldmodel/_version.py")
endif()
message(STATUS "hypercube_worldmodel version: ${HYPERCUBE_WORLDMODEL_VERSION}")

# ── Build the Python extension module ──
pybind11_add_module(_core bindings.cpp ${CORE_SOURCES})
target_include_directories(_core PRIVATE
    ${CORE_DIR}/Encoder
    ${CORE_DIR}/LCN
    ${CORE_DIR}/Predictor
    ${CORE_DIR}/Decoder
    ${CORE_DIR}/WorldModel)
target_compile_definitions(_core PRIVATE
    "HYPERCUBE_WORLDMODEL_VERSION=\"${HYPERCUBE_WORLDMODEL_VERSION}\"")

# ── Linking ──
if(MINGW)
    # Static libgcc/libstdc++ keep the .pyd free of those DLLs. Do NOT static-link
    # winpthread: mingw-w64 15.2.0's libwinpthread.a references __intrinsic_setjmpex
    # and fails at link (same constraint as the sibling python builds).
    # Posix-model MinGW still needs libwinpthread-1.dll at runtime; ship the DLL
    # from THIS toolchain (stale copies miss symbols like nanosleep64).
    target_link_options(_core PRIVATE -static-libgcc -static-libstdc++)
    find_file(WINPTHREAD_DLL libwinpthread-1.dll PATHS ENV PATH NO_DEFAULT_PATH)
    if(WINPTHREAD_DLL)
        install(FILES ${WINPTHREAD_DLL} DESTINATION hypercube_worldmodel)
    endif()
endif()

# ── Install into the hypercube_worldmodel package directory ──
install(TARGETS _core DESTINATION hypercube_worldmodel)
