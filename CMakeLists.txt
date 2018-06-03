cmake_minimum_required(VERSION 3.10)
project(neural_network)

set(ENABLE_CUDA 0)

if(${ENABLE_CUDA})

    FIND_PACKAGE(CUDA REQUIRED)

    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11;)

    include_directories(/usr/local/cuda/include)

    link_directories(/usr/local/cuda/lib64)

endif(${ENABLE_CUDA})

set(CMAKE_CXX_STANDARD 11)

ADD_DEFINITIONS(-O2)

set(
        SRC_LIST

        main.cpp
        v3/nn/layer/base/LayerBase.cpp
        v3/nn/layer/base/LayerBase.h
        v3/nn/optimizer/base/OptimizerBase.cpp
        v3/nn/optimizer/base/OptimizerBase.h
        v3/nn/layer/LinearLayer.cpp
        v3/nn/layer/LinearLayer.h
        v3/nn/layer/ConvLayer.cpp
        v3/nn/layer/ConvLayer.h
        v3/nn/layer/MaxPoolingLayer.cpp
        v3/nn/layer/MaxPoolingLayer.h
        v3/nn/layer/MeanPoolingLayer.cpp
        v3/nn/layer/MeanPoolingLayer.h
        v3/nn/layer/activation/SigmoidOutputLayer.cpp
        v3/nn/layer/activation/SigmoidOutputLayer.h
        v3/nn/layer/activation/ReLULayer.cpp
        v3/nn/layer/activation/ReLULayer.h
        v3/nn/layer/activation/LReLULayer.cpp
        v3/nn/layer/activation/LReLULayer.h
        v3/nn/layer/activation/SoftMaxOutputLayer.cpp
        v3/nn/layer/activation/SoftMaxOutputLayer.h
        v3/nn/optimizer/SGDOptimizer.cpp
        v3/nn/optimizer/SGDOptimizer.h
        v3/nn/optimizer/AdamOptimizer.cpp
        v3/nn/optimizer/AdamOptimizer.h
        v3/nn/optimizer/AdaMaxOptimizer.cpp
        v3/nn/optimizer/AdaMaxOptimizer.h
        v3/nn/NeuralNetwork.cpp
        v3/nn/NeuralNetwork.h
        v3/interface/interface.cpp
        v3/interface/interface.h
        v3/def/type.h
        v3/def/CUDAEnvironment.h
        v3/data/base/DataSetBase.cpp
        v3/data/base/DataSetBase.h
        v3/data/mnist/MNISTDataSet.cpp
        v3/data/mnist/MNISTDataSet.h
        v3/data/cifar10/CIFAR10DataSet.cpp
        v3/data/cifar10/CIFAR10DataSet.h
        v3/interface/interface.cu
        v3/utils/UniquePointerExt.h
        v3/utils/permutation.h
        v3/nn/layer/DropoutLayer.cpp
        v3/nn/layer/DropoutLayer.h
        v3/nn/layer/BatchNormLayer.cpp
        v3/nn/layer/BatchNormLayer.h
        v3/nn/AutoEncoder.cpp
        v3/nn/AutoEncoder.h)

if(${ENABLE_CUDA})

    cuda_add_executable(
            neural_network

            ${SRC_LIST}
    )

    target_link_libraries(neural_network cudart cublas)

else()

    add_executable(
            neural_network

            ${SRC_LIST}
    )

endif()