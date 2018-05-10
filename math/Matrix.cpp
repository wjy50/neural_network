/**
 * Created by wjy50 on 18-4-18.
 */

#include <cstring>
#include "../openblas/cblas.h"
#include "Matrix.h"

FloatType *multiplyMM(const FloatType *lhs, const FloatType *rhs, int x, int y, int z)
{
    auto *r = new FloatType[x * z];
    multiplyMMTo(r, lhs, rhs, x, y, z);
    return r;
}

void multiplyMMTo(FloatType *r, const FloatType *lhs, const FloatType *rhs, int x, int y, int z)
{
    memset(r, 0, x * z * sizeof(FloatType));
    for (int i = 0; i < x; ++i) {
        int r1 = i * z;
        for (int j = 0; j < y; ++j) {
            FloatType li = lhs[i*y+j];
            if (li != 0) {
                int r2 = j * z;
                for (int k = 0; k < z; ++k) {
                    r[r1+k] += li * rhs[r2+k];
                }
            }
        }
    }
}

FloatType *addMM(const FloatType *m1, const FloatType *m2, int x, int y)
{
    auto *r = new FloatType[x * y];
    addMMTo(r, m1, m2, x, y);
    return r;
}

void addMMTo(FloatType *r, const FloatType *m1, const FloatType *m2, int x, int y)
{
    int l = x * y;
    for (int i = 0; i < l; ++i) r[i] = m1[i] + m2[i];
}

FloatType *subMM(const FloatType *m1, const FloatType *m2, int x, int y)
{
    auto *r = new FloatType[x * y];
    subMMTo(r, m1, m2, x, y);
    return r;
}

void subMMTo(FloatType *r, const FloatType *m1, const FloatType *m2, int x, int y)
{
    int l = x * y;
    for (int i = 0; i < l; ++i) r[i] = m1[i] - m2[i];
}

FloatType *multiplyMV(const FloatType *m, const FloatType *v, int x, int y)
{
    auto *r = new FloatType[x];
    multiplyMVTo(r, m, v, x, y);
    return r;
}

FloatType *multiplyNM(FloatType n, const FloatType *m, int x, int y)
{
    auto *r = new FloatType[x * y];
    multiplyNMTo(r, n, m, x, y);
    return r;
}

void multiplyNMTo(FloatType *r, FloatType n, const FloatType *m, int x, int y)
{
    int l = x * y;
    for (int i = 0; i < l; ++i) {
        r[i] = m[i] * n;
    }
}

FloatType *addNM(FloatType n, const FloatType *m, int x, int y)
{
    auto *r = new FloatType[x * y];
    addNMTo(r, n, m, x, y);
    return r;
}

void addNMTo(FloatType *r, FloatType n, const FloatType *m, int x, int y)
{
    int l = x * y;
    for (int i = 0; i < l; ++i) {
        r[i] = m[i] + n;
    }
}

void mMultiplyN(FloatType n, FloatType *m, int x, int y)
{
    int l = x * y;
    for (int i = 0; i < l; ++i) {
        m[i] *= n;
    }
}

void mAddN(FloatType n, FloatType *m, int x, int y)
{
    int l = x * y;
    for (int i = 0; i < l; ++i) {
        m[i] += n;
    }
}

FloatType *multiplyMMElem(const FloatType *m1, const FloatType *m2, int x, int y)
{
    auto *r = new FloatType[x * y];
    multiplyMMElemTo(r, m1, m2, x, y);
    return r;
}

void multiplyMMElemTo(FloatType *r, const FloatType *m1, const FloatType *m2, int x, int y)
{
    int l = x * y;
    for (int i = 0; i < l; ++i) {
        r[i] = m1[i] * m2[i];
    }
}

FloatType *transposeM(const FloatType *m, int x, int y)
{
    auto *r = new FloatType[y * x];
    transposeMTo(r, m, x, y);
    return r;
}

void transposeMTo(FloatType *r, const FloatType *m, int x, int y)
{
    for (int i = 0; i < x; ++i) {
        int mOffset = i * y;
        for (int j = 0; j < y; ++j) {
            r[j*x+i] = m[mOffset + j];
        }
    }
}

FloatType *multiplyTransposedMV(const FloatType *m, const FloatType *v, int x, int y)
{
    auto *r = new FloatType[y];
    multiplyTransposedMVTo(r, m, v, x, y);
    return r;
}

void multiplyTransposedMVTo(FloatType *r, const FloatType *m, const FloatType *v, int x, int y)
{
    memset(r, 0, y * sizeof(FloatType));
    for (int i = 0; i < x; ++i) {
        FloatType vi = v[i];
        if (vi != 0) {
            const FloatType *mi = m + i * y;
            for (int j = 0; j < y; ++j) {
                r[j] += mi[j] * vi;
            }
        }
    }
}

#if !USE_CBLAS

void multiplyMVTo(FloatType *r, const FloatType *m, const FloatType *v, int x, int y)
{
    for (int i = 0; i < x; ++i) {
        FloatType res = 0;
        const FloatType *mi = m + i * y;
        for (int j = 0; j < y; ++j) {
            res += mi[j] * v[j];
        }
        r[i] = res;
    }
}

#endif