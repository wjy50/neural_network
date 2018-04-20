#include <iostream>
#include <memory>
#include "math/Matrix.h"
#include "mnist/mnist.h"
#include "nn/NeuralNetwork.h"
#include "math/Activator.h"
#include "math/permutation.h"

using namespace std;

int main()
{
    std::cout << "Hello, World!" << std::endl;
    int nums[] = {28*28, 100, 80, 10};
    NeuralNetwork network(nums, 4, lReLU, dLReLU_dx);
    network.initialize();
    MNISTImage image("/home/wjy50/mnist/train-images.idx3-ubyte");
    MNISTLabel label("/home/wjy50/mnist/train-labels.idx1-ubyte");
    MNISTImage testImage("/home/wjy50/mnist/t10k-images.idx3-ubyte");
    MNISTLabel testLabel("/home/wjy50/mnist/t10k-labels.idx1-ubyte");
    unique_ptr<int> indices(new int[50000]);
    for (int k = 0; k < 30; ++k) {
        network.SGD(image, label, 50000, 10);
        /*randomPermutation(indices.get(), 50000);
        for (int i = 0; i < 50000; ++i) {
            network.train(image.get(indices.get()[i]), label.get(indices.get()[i]));
        }*/
        int fail = 0;
        for (int j = 0; j < 10000; ++j) {
            if (!network.test(image.get(50000+j), label.get(50000+j))) fail++;
        }
        cout << "epoch" << k+1 << ':' << fail << endl;
    }
    int fail = 0;
    for (int j = 0; j < 10000; ++j) {
        if (!network.test(testImage.get(j), testLabel.get(j))) fail++;
    }
    cout << fail << endl;
    return 0;
}