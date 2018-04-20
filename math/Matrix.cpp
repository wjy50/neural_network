/**
 * Created by wjy50 on 18-4-18.
 */

#include "Matrix.h"

double *multiplyMM(const double *lhs, const double *rhs, int x, int y, int z)
{
    auto *r = new double[x*z];
    multiplyMMTo(r, lhs, rhs, x, y, z);
    return r;
}

void multiplyMMTo(double *r, const double *lhs, const double *rhs, int x, int y, int z)
{
    for (int i = 0; i < x * z; ++i) {
        r[i] = 0;
    }
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            if (lhs[i*y+j] != 0) for (int k = 0; k < z; ++k) {
                    r[i*z+k] += lhs[i*y+j] * rhs[j*z+k];
            }
        }
    }
}

double *addMM(const double *m1, const double *m2, int x, int y)
{
    auto *r = new double[x*y];
    addMMTo(r, m1, m2, x, y);
    return r;
}

void addMMTo(double *r, const double *m1, const double *m2, int x, int y)
{
    for (int i = 0; i < x * y; ++i) r[i] = m1[i] + m2[i];
}

double *subMM(const double *m1, const double *m2, int x, int y)
{
    auto *r = new double[x*y];
    subMMTo(r, m1, m2, x, y);
    return r;
}

void subMMTo(double *r, const double *m1, const double *m2, int x, int y)
{
    for (int i = 0; i < x * y; ++i) r[i] = m1[i] - m2[i];
}

double *multiplyMV(const double *m, const double *v, int x, int y)
{
    auto *r = new double[x];
    multiplyMVTo(r, m, v, x, y);
    return r;
}

void multiplyMVTo(double *r, const double *m, const double *v, int x, int y)
{
    for (int i = 0; i < x; ++i) {
        r[i] = 0;
        for (int j = 0; j < y; ++j) {
            r[i] += m[i*y+j]*v[j];
        }
    }
}

double *multiplyNM(double n, const double *m, int x, int y)
{
    auto *r = new double[x*y];
    multiplyNMTo(r, n, m, x, y);
    return r;
}

void multiplyNMTo(double *r, double n, const double *m, int x, int y)
{
    for (int i = 0; i < x * y; ++i) {
        r[i] = m[i] * n;
    }
}

double *addNM(double n, const double *m, int x, int y)
{
    auto *r = new double[x*y];
    addNMTo(r, n, m, x, y);
    return r;
}

void addNMTo(double *r, double n, const double *m, int x, int y)
{
    for (int i = 0; i < x * y; ++i) {
        r[i] = m[i] + n;
    }
}

void mMultiplyN(double n, double *m, int x, int y)
{
    for (int i = 0; i < x * y; ++i) {
        m[i] *= n;
    }
}

void mAddN(double n, double *m, int x, int y)
{
    for (int i = 0; i < x * y; ++i) {
        m[i] += n;
    }
}

double *multiplyMMElem(const double *m1, const double *m2, int x, int y)
{
    auto *r = new double[x*y];
    multiplyMMElemTo(r, m1, m2, x, y);
    return r;
}

void multiplyMMElemTo(double *r, const double *m1, const double *m2, int x, int y)
{
    for (int i = 0; i < x * y; ++i) {
        r[i] = m1[i] * m2[i];
    }
}

double *transposeM(const double *m, int x, int y)
{
    auto *r = new double[y*x];
    transposeMTo(r, m, x, y);
    return r;
}

void transposeMTo(double *r, const double *m, int x, int y)
{
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            r[j*x+i] = m[i*y+j];
        }
    }
}