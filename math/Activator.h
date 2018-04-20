/**
 * Created by wjy50 on 18-4-16.
 */

#ifndef NEURAL_NETWORK_ACTIVATOR_H
#define NEURAL_NETWORK_ACTIVATOR_H

#include <cmath>

#ifdef __cplusplus
extern "C"
{
#endif

    double sigmoid(double x) { return 1/(1+std::exp(-x)); }

    double dSigmoid_dx(double x)
    {
        double sig = 1/(1+std::exp(-x)); /*sigmoid(x)*/
        return sig*(1-sig);
    }

    double ReLU(double x) { return std::fmax(0, x); }

    double dReLU_dx(double x) { return x > 0 ? 1 : 0; }

    double lReLU(double x) { return x > 0 ? x : 0.1*x; }

    double dLReLU_dx(double x) { return x > 0 ? 1 : 0.1; }

#ifdef __cplusplus
};
#endif

#endif //NEURAL_NETWORK_ACTIVATOR_H
