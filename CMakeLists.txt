cmake_minimum_required(VERSION 3.10)
project(neural_network)

set(CMAKE_CXX_STANDARD 11)

add_executable(neural_network main.cpp nn/NeuralNetwork.cpp math/Matrix.h math/Activator.h math/Matrix.cpp mnist/mnist.h mnist/mnist.cpp math/permutation.h math/permutation.cpp math/Activator.cpp)