/**
 * Created by wjy50 on 18-4-21.
 */

#include <cmath>
#include <cstddef>
#include "Activator.h"

FloatType sigmoid(FloatType x) { return 1/(1+std::exp(-x)); }

FloatType dSigmoid_dx(FloatType x)
{
    FloatType sig = 1/(1+std::exp(-x)); /*sigmoid(x)*/
    return sig*(1-sig);
}

FloatType ReLU(FloatType x) { return std::fmax(static_cast<FloatType>(0), x); }

FloatType dReLU_dx(FloatType x) { return x > 0 ? 1 : 0; }

FloatType lReLU(FloatType x) { return x > 0 ? x : static_cast<FloatType>(0.01)*x; }

FloatType dLReLU_dx(FloatType x) { return x > 0 ? 1 : static_cast<FloatType>(0.01); }

void softMax(FloatType *x, int n)
{
    FloatType m = 0;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i]);
        m += x[i];
    }
    for (int i = 0; i < n; ++i) {
        x[i] /= m;
    }
}

void softMaxInto(FloatType *r, FloatType *x, int n)
{
    FloatType m = 0;
    for (int i = 0; i < n; ++i) {
        r[i] = std::exp(x[i]);
        m += r[i];
    }
    for (int i = 0; i < n; ++i) {
        r[i] /= m;
    }
}