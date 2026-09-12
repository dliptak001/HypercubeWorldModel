cmake_minimum_required(VERSION 3.21)
project(HypercubeWorldModel VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Family convention (matches HypercubeWTF / HypercubeLCN): fast-math in Release.
add_compile_options(
    "$<$<CONFIG:Release>:-O3;-march=x86-64-v2;-ffast-math>"
    "$<$<CONFIG:Debug>:-O0;-g>"
    -Wall -Wextra)

# Encoder: frozen hypercube reservoir (HypercubeWTF episode)
add_library(Encoder STATIC
    Encoder/Encoder.cpp)
target_include_directories(Encoder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Encoder)

# LCN: locally connected net + trainer (port of HypercubeLCN Core / Training;
# weight layout is depth, axis, tap, vertex, not HypercubeLCN's vertex-major)
add_library(LCN STATIC
    LCN/LCN.cpp
    LCN/LCNTraining.cpp)
target_include_directories(LCN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/LCN)

# ThreadPool: persistent std::thread workers (no OpenMP)
add_library(ThreadPool STATIC
    ThreadPool/ThreadPool.cpp)
target_include_directories(ThreadPool PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ThreadPool)
if(MINGW)
    target_link_libraries(ThreadPool PUBLIC pthread)
endif()

# Decoder: invert an encoder subcube to the full cube
add_library(Decoder STATIC
    Decoder/Decoder.cpp)
target_include_directories(Decoder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Decoder)
target_link_libraries(Decoder PUBLIC LCN)

# Predictor: k-face → predicted next k-face
add_library(Predictor STATIC
    Predictor/Predictor.cpp)
target_include_directories(Predictor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Predictor)
target_link_libraries(Predictor PUBLIC LCN)

# WorldModel: frozen Encoder + trained Predictor; does the k-cut
add_library(WorldModel STATIC
    WorldModel/WorldModel.cpp)
target_include_directories(WorldModel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/WorldModel)
target_link_libraries(WorldModel PUBLIC Encoder Predictor)

# VectorModel: optional third product library for short-vector hosts.
# Normaliser, Head, MinDim/MinK, and the MSE/linR2 metrics live here.
add_library(VectorModel STATIC
    Normaliser/Normaliser.cpp
    Head/Solver.cpp
    Head/Head.cpp
    VectorModel/VectorModel.cpp
    VectorModel/Metrics.cpp)
target_include_directories(VectorModel PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Normaliser
    ${CMAKE_CURRENT_SOURCE_DIR}/Head
    ${CMAKE_CURRENT_SOURCE_DIR}/VectorModel)
target_link_libraries(VectorModel PUBLIC WorldModel)

add_executable(HypercubeWorldModel main.cpp)
target_link_libraries(HypercubeWorldModel PRIVATE Encoder LCN Decoder Predictor WorldModel VectorModel)

# SDK quick start: the program in docs/CPP_SDK.md, compiled so the doc stays true
add_executable(quick_start examples/quick_start.cpp)
target_link_libraries(quick_start PRIVATE WorldModel)

add_executable(vector_start examples/vector_start.cpp)
target_link_libraries(vector_start PRIVATE VectorModel)

if(MINGW)
    target_link_options(HypercubeWorldModel PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(quick_start PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(vector_start PRIVATE -static-libgcc -static-libstdc++)
endif()
