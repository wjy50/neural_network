cmake_minimum_required(VERSION 3.10)
project(neural_network)

set(CMAKE_CXX_STANDARD 11)

add_executable(neural_network main.cpp nn/NeuralNetwork.cpp math/Matrix.h math/Activator.h math/Matrix.cpp mnist/mnist.h mnist/mnist.cpp math/permutation.h math/Activator.cpp utils/UniquePointerExt.h nn/ffw/FeedForwardNN.h nn/ffw/FeedForwardNN.cpp nn/ffw/layer/AbsLayer.cpp nn/ffw/layer/AbsLayer.h nn/ffw/layer/FullyConnLayer.cpp nn/ffw/layer/FullyConnLayer.h nn/ffw/layer/ConvLayer.cpp nn/ffw/layer/ConvLayer.h nn/ffw/layer/PoolingLayer.cpp nn/ffw/layer/PoolingLayer.h nn/ffw/layer/BatchNormLayer.cpp nn/ffw/layer/BatchNormLayer.h data/DataSet.h)