/**
 * Created by wjy50 on 18-4-21.
 */

#include <cmath>
#include "Activator.h"

double sigmoid(double x) { return 1/(1+exp(-x)); }

double dSigmoid_dx(double x)
{
    double sig = 1/(1+exp(-x)); /*sigmoid(x)*/
    return sig*(1-sig);
}

double ReLU(double x) { return fmax(0, x); }

double dReLU_dx(double x) { return x > 0 ? 1 : 0; }

double lReLU(double x) { return x > 0 ? x : 0.1*x; }

double dLReLU_dx(double x) { return x > 0 ? 1 : 0.1; }

void softMax(double *x, int n)
{
    double m = 0;
    for (int i = 0; i < n; ++i) {
        x[i] = exp(x[i]);
        m += x[i];
    }
    for (int i = 0; i < n; ++i) {
        x[i] /= m;
    }
}

void softMaxInto(double *r, double *x, int n)
{
    double m = 0;
    for (int i = 0; i < n; ++i) {
        r[i] = exp(x[i]);
        m += r[i];
    }
    for (int i = 0; i < n; ++i) {
        r[i] /= m;
    }
}