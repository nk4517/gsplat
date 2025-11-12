cmake_minimum_required(VERSION 3.28)
project(gsplat_2025 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 89)

# CUDA compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

include_directories(gsplat/cuda/csrc)
include_directories(gsplat/cuda/csrc/third_party/glm/)
include_directories(gsplat/cuda/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(C:/Users/user/AppData/Local/pypoetry/Cache/virtualenvs/gsplat-sandbox-2l1tlsyc-py3.10/Lib/site-packages/torch/include)

add_library(gsplat_2025 STATIC
        gsplat/cuda/csrc/Adam.cpp
        gsplat/cuda/csrc/Adam.h
        gsplat/cuda/csrc/AdamCUDA.cu
        gsplat/cuda/csrc/Intersect.cpp
        gsplat/cuda/csrc/IntersectTile.cu
        gsplat/cuda/csrc/Null.cpp
        gsplat/cuda/csrc/NullCUDA.cu
        gsplat/cuda/csrc/Projection.cpp
        gsplat/cuda/csrc/Projection2DGSFused.cu
        gsplat/cuda/csrc/Projection2DGSPacked.cu
        gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu
        gsplat/cuda/csrc/ProjectionEWA3DGSPacked.cu
        gsplat/cuda/csrc/ProjectionEWASimple.cu
        gsplat/cuda/csrc/ProjectionUT3DGSFused.cu
        gsplat/cuda/csrc/QuatScaleToCovar.cpp
        gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu
        gsplat/cuda/csrc/Rasterization.cpp
        gsplat/cuda/csrc/RasterizeToIndices2DGS.cu
        gsplat/cuda/csrc/RasterizeToIndices3DGS.cu
        gsplat/cuda/csrc/RasterizeToPixels2DGSBwd.cu
        gsplat/cuda/csrc/RasterizeToPixels2DGSFwd.cu
        gsplat/cuda/csrc/RasterizeToPixels2DGSWSumBwd.cu
        gsplat/cuda/csrc/RasterizeToPixels2DGSWSumFwd.cu
        gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu
        gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu
        gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu
        gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSFwd.cu
        gsplat/cuda/csrc/Relocation.cpp
        gsplat/cuda/csrc/RelocationCUDA.cu
        gsplat/cuda/csrc/SphericalHarmonics.cpp
        gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu
        gsplat/cuda/ext.cpp)
