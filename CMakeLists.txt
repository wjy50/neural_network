cmake_minimum_required(VERSION 3.10)
project(neural_network)

set(CMAKE_CXX_STANDARD 11)

ADD_DEFINITIONS(-O3)

add_library(
        openblas

        SHARED

        IMPORTED
)

set_target_properties(
        openblas

        PROPERTIES

        IMPORTED_LOCATION

        ${CMAKE_SOURCE_DIR}/libopenblas.so
)

add_library(
        openblas_haswell

        SHARED

        IMPORTED
)

set_target_properties(
        openblas_haswell

        PROPERTIES

        IMPORTED_LOCATION

        ${CMAKE_SOURCE_DIR}/libopenblas_haswellp-r0.2.20.so
)

add_executable(
        neural_network
        main.cpp
        math/Matrix.h
        math/Activator.h
        math/Matrix.cpp
        mnist/mnist.h
        mnist/mnist.cpp
        math/permutation.h
        math/Activator.cpp
        utils/UniquePointerExt.h
        nn/ffw/FeedForwardNN.h
        nn/ffw/FeedForwardNN.cpp
        nn/ffw/layer/AbsLayer.cpp
        nn/ffw/layer/AbsLayer.h
        nn/ffw/layer/FullyConnLayer.cpp
        nn/ffw/layer/FullyConnLayer.h
        nn/ffw/layer/ConvLayer.cpp
        nn/ffw/layer/ConvLayer.h
        nn/ffw/layer/PoolingLayer.cpp
        nn/ffw/layer/PoolingLayer.h
        nn/ffw/layer/BatchNormLayer.cpp
        nn/ffw/layer/BatchNormLayer.h
        data/DataSet.h cifar/cifar.h
        cifar/cifar.cpp
        data/DataSet.cpp
        math/mtype.h
        nn/ffw/optimizer/AbsOptimizer.cpp
        nn/ffw/optimizer/AbsOptimizer.h
        nn/ffw/optimizer/SimpleSGDOptimizer.cpp
        nn/ffw/optimizer/SimpleSGDOptimizer.h
        nn/ffw/optimizer/AdamOptimizer.cpp
        nn/ffw/optimizer/AdamOptimizer.h
        nn/ffw/optimizer/AdaMaxOptimizer.cpp
        nn/ffw/optimizer/AdaMaxOptimizer.h
        openblas/cblas.h
        openblas/f77blas.h
)

target_link_libraries(neural_network openblas openblas_haswell)