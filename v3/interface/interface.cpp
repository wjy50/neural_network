/**
 * Created by wjy50 on 18-5-18.
 */

#include <cmath>
#include <algorithm>
#include "interface.h"

void freeArray(void *p)
{
#if ENABLE_CUDA
    cudaFree(p);
#else
    delete[] p;
#endif
}

#if !ENABLE_CUDA

void leakyReLU(FloatType *out, const FloatType *in, int len, FloatType l)
{
    for (int i = 0; i < len; ++i) {
        out[i] = in[i] >= 0 ? in[i] : (l * in[i]);
    }
}

void leakyReLU_bp(FloatType *out, const FloatType *x, const FloatType *delta, int len, FloatType l)
{
    for (int i = 0; i < len; ++i) {
        out[i] = x[i] >= 0 ? delta[i] : (delta[i] * l);
    }
}

void reLU(FloatType *out, const FloatType *in, int len)
{
    for (int i = 0; i < len; ++i) {
        out[i] = in[i] >= 0 ? in[i] : 0;
    }
}

void reLU_bp(FloatType *out, const FloatType *x, const FloatType *delta, int len)
{
    for (int i = 0; i < len; ++i) {
        out[i] = x[i] >= 0 ? delta[i] : 0;
    }
}

void sigmoidOutput(FloatType *out, const FloatType *in, int len)
{
    for (int i = 0; i < len; ++i) {
        out[i] = 1 / (1 + std::exp(-in[i]));
    }
}

void addVTo(FloatType *r, const FloatType *a, const FloatType *b, int len)
{
    for (int i = 0; i < len; ++i) {
        r[i] = a[i] + b[i];
    }
}

void subtractVTo(FloatType *r, const FloatType *a, const FloatType *b, int len)
{
    for (int i = 0; i < len; ++i) {
        r[i] = a[i] - b[i];
    }
}

void softMaxOutput(FloatType *out, const FloatType *in, int len, int count)
{
    for (int i = 0; i < count; ++i) {
        FloatType sum = 0;
        for (int j = 0; j < len; ++j) {
            out[j] = std::exp(in[j]);
            sum += out[j];
        }
        for (int j = 0; j < len; ++j) {
            out[j] /= sum;
        }
        out += len;
        in += len;
    }
}

void sgd(FloatType *params, const FloatType *gradients, FloatType eta, int len)
{
    for (int i = 0; i < len; ++i) {
        params[i] += eta * gradients[i];
    }
}

void l2SGD(FloatType *params, const FloatType *gradients, FloatType eta, FloatType reg, int len)
{
    for (int i = 0; i < len; ++i) {
        params[i] = params[i] * reg + eta * gradients[i];
    }
}

void adamFirstMomentEstimate(FloatType *m, FloatType beta, FloatType oneMBeta, const FloatType *g, int len)
{
    for (int i = 0; i < len; ++i) {
        m[i] = beta * m[i] + oneMBeta * g[i];
    }
}

void adamSecondMomentEstimate(FloatType *v, FloatType beta, FloatType oneMBeta, const FloatType *g, int len)
{
    for (int i = 0; i < len; ++i) {
        v[i] = beta * v[i] + oneMBeta * g[i] * g[i];
    }
}

void adamUpdate(FloatType *params, const FloatType *m, const FloatType *v, int len, FloatType alpha,
                FloatType oneMBeta1T, FloatType oneMBeta2T)
{
    for (int i = 0; i < len; ++i) {
        params[i] += alpha * m[i] / (oneMBeta1T * std::sqrt(v[i] / oneMBeta2T) + 1e-6);
    }
}

void adaMaxEWIN(FloatType *u, FloatType beta, const FloatType *g, int len)
{
    for (int i = 0; i < len; ++i) {
        u[i] = std::fmax(beta * u[i], std::fabs(g[i]));
    }
}

void adaMaxUpdate(FloatType *params, const FloatType *m, const FloatType *u, int len,
                  FloatType learningRate, FloatType betaTMOne)
{
    for (int i = 0; i < len; ++i) {
        params[i] += learningRate * m[i] / (betaTMOne * u[i] + 1e-6);
    }
}

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

void multiplyMMTo(FloatType *r, const FloatType *lhs, const FloatType *rhs, int x, int y, int z)
{
    memset(r, 0, x * z * sizeof(FloatType));
    for (int i = 0; i < x; ++i) {
        int r1 = i * z;
        for (int j = 0; j < y; ++j) {
            const FloatType li = lhs[i * y + j];
            if (li != 0) {
                int r2 = j * z;
                for (int k = 0; k < z; ++k) {
                    r[r1 + k] += li * rhs[r2 + k];
                }
            }
        }
    }
}

void multiplyMTmTo(FloatType *r, const FloatType *lhs, const FloatType *rhs, int x, int y, int z)
{
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < z; ++j) {
            FloatType sum = 0;
            for (int k = 0; k < y; ++k) {
                sum += lhs[i * y + k] * rhs[j * y + k];
            }
            r[i * z + j] = sum;
        }
    }
}

void multiplyTmMTo(FloatType *r, const FloatType *lhs, const FloatType *rhs, int x, int y, int z)
{
    memset(r, 0, y * z * sizeof(FloatType));
    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            if (lhs[i * y + j] != 0) for (int k = 0; k < z; ++k) {
                r[j * z + k] += lhs[i * y + j] * rhs[i * z + k];
            }
        }
    }
}

void multiplyNVTo(FloatType *r, FloatType n, const FloatType *v, int dim)
{
    for (int i = 0; i < dim; ++i) {
        r[i] = v[i] * n;
    }
}

void averageVTo(FloatType *r, const FloatType *v, int dim, int count)
{
    memset(r, 0, dim * sizeof(FloatType));
    for (int i = 0; i < count; ++i) {
        const FloatType *cv = v + dim * i;
        for (int j = 0; j < dim; ++j) {
            r[j] += cv[j];
        }
    }
    for (int i = 0; i < dim; ++i) {
        r[i] /= count;
    }
}

/*void varianceVTo(FloatType *r, const FloatType *v, const FloatType *avg, int dim, int count)
{
    memset(r, 0, dim * sizeof(FloatType));
    for (int i = 0; i < count; ++i) {
        const FloatType *cv = v + i * dim;
        for (int j = 0; j < dim; ++j) {
            const FloatType err = cv[j] - avg[j];
            r[j] += err * err;
        }
    }
    for (int i = 0; i < dim; ++i) {
        r[i] /= count;
    }
}*/

void meanPooling(const FloatType *in, int inputWidth, int inputHeight,
                 FloatType *out, int outputWidth, int outputHeight,
                 int windowWidth, int windowHeight,
                 int xStride, int yStride,
                 int channelCount
)
{
    int inputSize = inputWidth * inputHeight;
    int outputSize = outputWidth * outputHeight;
    int windowSize = windowWidth * windowHeight;
    for (int i = 0; i < channelCount; ++i) {
        const FloatType *curIn = in + i * inputSize;
        FloatType *curOut = out + i * outputSize;
        int oy = 0;
        for (int j = 0; j <= inputHeight - windowHeight; j += yStride, ++oy) {
            int ox = 0;
            for (int k = 0; k <= inputWidth - windowWidth; k += xStride, ++ox) {
                FloatType sum = 0;
                for (int l = 0; l < windowHeight; ++l) {
                    for (int m = 0; m < windowWidth; ++m) {
                        sum += curIn[(j + l) * inputWidth + k + m];
                    }
                }
                curOut[oy * outputWidth + ox] = sum / windowSize;
            }
        }
    }
}

int indexOfMax(const FloatType *arr, int n)
{
    int maxI = 0;
    FloatType max = arr[0];
    for (int i = 1; i < n; ++i) {
        if (arr[i] > max) {
            maxI = i;
            max = arr[i];
        }
    }
    return maxI;
}

void maxPooling(const FloatType *in, int inputWidth, int inputHeight,
                FloatType *out, int outputWidth, int outputHeight,
                int *xOffset, int *yOffset,
                int windowWidth, int windowHeight,
                int xStride, int yStride,
                int channelCount
)
{
    int inputSize = inputWidth * inputHeight;
    int outputSize = outputWidth * outputHeight;
    if (xOffset && yOffset) {
        for (int i = 0; i < channelCount; ++i) {
            const FloatType *curIn = in + i * inputSize;
            FloatType *curOut = out + i * outputSize;
            int *xo = xOffset + i * outputSize;
            int *yo = yOffset + i * outputSize;
            int oy = 0;
            for (int j = 0; j <= inputHeight - windowHeight; j += yStride, ++oy) {
                int ox = 0;
                for (int k = 0; k <= inputWidth - windowWidth; k += xStride, ++ox) {
                    int maxR = 0;
                    int maxC = indexOfMax(curIn + j * inputWidth + k, windowWidth);
                    FloatType max = curIn[j * inputWidth + k + maxC];
                    for (int l = 1; l < windowHeight; ++l) {
                        int c = indexOfMax(curIn + (j + l) * inputWidth + k, windowWidth);
                        if (curIn[(j + l) * inputWidth + k + c] > max) {
                            max = curIn[(j + l) * inputWidth + k + c];
                            maxR = l;
                            maxC = c;
                        }
                    }
                    const int o = oy * outputWidth + ox;
                    curOut[o] = max;
                    xo[o] = maxC;
                    yo[o] = maxR;
                }
            }
        }
    } else {
        for (int i = 0; i < channelCount; ++i) {
            const FloatType *curIn = in + i * inputSize;
            FloatType *curOut = out + i * outputSize;
            int oy = 0;
            for (int j = 0; j <= inputHeight - windowHeight; j += yStride, ++oy) {
                int ox = 0;
                for (int k = 0; k <= inputWidth - windowWidth; k += xStride, ++ox) {
                    int c = indexOfMax(curIn + j * inputWidth + k, windowWidth);
                    FloatType max = curIn[j * inputWidth + k + c];
                    for (int l = 1; l < windowHeight; ++l) {
                        c = indexOfMax(curIn + (j + l) * inputWidth + k, windowWidth);
                        if (curIn[(j + l) * inputWidth + k + c] > max) {
                            max = curIn[(j + l) * inputWidth + k + c];
                        }
                    }
                    curOut[oy * outputWidth + ox] = max;
                }
            }
        }
    }
}

void meanPoolingBP(const FloatType *in, int inputWidth, int inputHeight,
                   FloatType *out, int outputWidth, int outputHeight,
                   int windowWidth, int windowHeight,
                   int xStride, int yStride,
                   int channelCount
)
{
    int outputSize = outputWidth * outputHeight;
    int inputSize = inputWidth * inputHeight;
    int windowSize = windowWidth * windowHeight;
    memset(out, 0, outputSize * channelCount * sizeof(FloatType));
    for (int i = 0; i < channelCount; ++i) {
        FloatType *od = out + i * outputSize;
        const FloatType *d = in + i * inputSize;
        int oy = 0;
        for (int j = 0; j <= outputHeight - windowHeight; j += yStride, ++oy) {
            int ox = 0;
            for (int k = 0; k <= outputWidth - windowWidth; k += xStride, ++ox) {
                const FloatType di = d[oy * inputWidth + ox] / windowSize;
                for (int l = 0; l < windowHeight; ++l) {
                    for (int m = 0; m < windowWidth; ++m) {
                        od[(j + l) * outputWidth + k + m] += di;
                    }
                }
            }
        }
    }
}

void maxPoolingBP(const FloatType *in, int inputWidth, int inputHeight,
                  FloatType *out, int outputWidth, int outputHeight,
                  const int *xOffset, const int *yOffset,
                  int windowWidth, int windowHeight,
                  int xStride, int yStride,
                  int channelCount
)
{
    int inputSize = inputWidth * inputHeight;
    int outputSize = outputWidth * outputHeight;
    memset(out, 0, outputSize * channelCount * sizeof(FloatType));
    for (int i = 0; i < channelCount; ++i) {
        FloatType *od = out + i * outputSize;
        const FloatType *d = in + i * inputSize;
        const int *xo = xOffset + i * inputSize;
        const int *yo = yOffset + i * inputSize;
        int oy = 0;
        for (int j = 0; j <= outputHeight - windowHeight; j += yStride, ++oy) {
            int ox = 0;
            for (int k = 0; k <= outputWidth - windowWidth; k += xStride, ++ox) {
                const int o = oy * inputWidth + ox;
                od[(j + yo[o]) * outputWidth + k + xo[o]] += d[o];
            }
        }
    }
}

void doConv(const FloatType *in, int inputWidth, int inputHeight,
            FloatType *out, int outputWidth,
            const FloatType *kernel, int kernelWidth, int kernelHeight,
            int xStride, int yStride, int xPadding, int yPadding
)
{
    int bottom = inputHeight + yPadding - kernelHeight;
    int right = inputWidth + xPadding - kernelWidth;
    for (int kerY = 0; kerY < kernelHeight; ++kerY) {
        const FloatType *kerRow = kernel + kerY * kernelWidth;
        int inputY = kerY - yPadding;
        int oy = 0;
        while (inputY < 0) {
            inputY += yStride;
            oy++;
        }
        int yLim = std::min(bottom + kerY, inputHeight - 1);
        for (; inputY <= yLim; inputY += yStride, oy++) {
            const FloatType *inputRow = in + inputY * inputWidth;
            FloatType *outputRow = out + oy * outputWidth;
            int ox = 0;
            for (int inputX = -yPadding; inputX <= right; inputX += xStride, ++ox) {
                int xLim = kernelWidth - std::max(0, inputX + kernelWidth - inputWidth);
                FloatType sum = 0;
                for (int kerX = std::max(-inputX, 0); kerX < xLim; ++kerX) {
                    sum += kerRow[kerX] * inputRow[inputX + kerX];
                }
                outputRow[ox] += sum;
            }
        }
    }
}

void conv(const FloatType *in, int inputWidth, int inputHeight, int channelCount,
          FloatType *out, int outputWidth, int outputHeight,
          const FloatType *kernel, int kernelWidth, int kernelHeight, int kernelCount,
          int xStride, int yStride, int xPadding, int yPadding,
          int batchSize
)
{
    int inputSize = inputWidth * inputHeight;
    int outputSize = outputWidth * outputHeight;
    int kernelSize = kernelWidth * kernelHeight;
    memset(out, 0, outputSize * kernelCount * batchSize * sizeof(FloatType));
    for (int i = 0; i < batchSize; ++i) {
        const FloatType *currentInput = in + i * inputSize * channelCount;
        FloatType *currentOutput = out + i * outputSize * kernelCount;
        for (int j = 0; j < kernelCount; ++j) {
            FloatType *outputLayer = currentOutput + j * outputSize;
            const FloatType *currentKernel = kernel + j * kernelSize * channelCount;
            for (int k = 0; k < channelCount; ++k) {
                doConv(currentInput + k * inputSize, inputWidth, inputHeight,
                       outputLayer, outputWidth,
                       currentKernel + k * kernelSize, kernelWidth, kernelHeight,
                       xStride, yStride, xPadding, yPadding
                );
            }
        }
    }
}

void doConvBP(const FloatType *in, int inputWidth,
              FloatType *out, int outputWidth, int outputHeight,
              const FloatType *kernel, int kernelWidth, int kernelHeight,
              int xStride, int yStride, int xPadding, int yPadding
)
{
    int bottom = outputHeight + yPadding - kernelHeight;
    int right = outputWidth + xPadding - kernelWidth;
    for (int kerY = 0; kerY < kernelHeight; ++kerY) {
        const FloatType *kerRow = kernel + kerY * kernelWidth;
        int outputY = kerY - yPadding;
        int iy = 0;
        while (outputY < 0) {
            outputY += yStride;
            iy++;
        }
        int yLim = std::min(bottom + kerY, outputHeight - 1);
        for (; outputY <= yLim; outputY += yStride, iy++) {
            FloatType *outputRow = out + outputY * outputWidth;
            const FloatType *inputRow = in + iy * inputWidth;
            int ix = 0;
            for (int outputX = -yPadding; outputX <= right; outputX += xStride, ++ix) {
                int xLim = kernelWidth - std::max(0, outputX + kernelWidth - outputWidth);
                FloatType d = inputRow[ix];
                for (int kerX = std::max(-outputX, 0); kerX < xLim; ++kerX) {
                    outputRow[outputX + kerX] += d * kerRow[kerX];
                }
            }
        }
    }
}

void convBP(const FloatType *in, int inputWidth, int inputHeight,
            FloatType *out, int outputWidth, int outputHeight, int channelCount,
            const FloatType *kernel, int kernelWidth, int kernelHeight, int kernelCount,
            int xStride, int yStride, int xPadding, int yPadding,
            int batchSize
)
{
    int inputSize = inputWidth * inputHeight;
    int outputSize = outputWidth * outputHeight;
    int kernelSize = kernelWidth * kernelHeight;
    memset(out, 0, outputSize * channelCount * batchSize * sizeof(FloatType));
    for (int i = 0; i < batchSize; ++i) {
        const FloatType *currentInput = in + i * inputSize * kernelCount;
        FloatType *currentOutput = out + i * outputSize * channelCount;
        for (int j = 0; j < kernelCount; ++j) {
            const FloatType *inputLayer = currentInput + j * inputSize;
            const FloatType *currentKernel = kernel + j * kernelSize * channelCount;
            for (int k = 0; k < channelCount; ++k) {
                doConvBP(inputLayer, inputWidth,
                         currentOutput + k * outputSize, outputWidth, outputHeight,
                         currentKernel + k * kernelSize, kernelWidth, kernelHeight,
                         xStride, yStride, xPadding, yPadding
                );
            }
        }
    }
}

void convGradients(FloatType *kernel, int kernelWidth, int kernelHeight,
                   FloatType *biases, int kernelCount,
                   const FloatType *delta, int outputWidth, int outputHeight,
                   const FloatType *input, int inputWidth, int inputHeight, int channelCount,
                   int xStride, int yStride, int xPadding, int yPadding,
                   int batchSize
)
{
    int inputSize = inputWidth * inputHeight;
    int outputSize = outputWidth * outputHeight;
    int kernelSize = kernelWidth * kernelHeight;
    int weightCount = kernelSize * channelCount * kernelCount;
    memset(kernel, 0, weightCount * sizeof(FloatType));
    for (int i = 0; i < batchSize; ++i) {
        const FloatType *currentInput = input + i * inputSize * channelCount;
        const FloatType *currentOutput = delta + i * outputSize * kernelCount;
        for (int j = 0; j < kernelCount; ++j) {
            const FloatType *outputLayer = currentOutput + j * outputSize;
            FloatType *currentKernel = kernel + j * kernelSize * channelCount;
            for (int k = 0; k < channelCount; ++k) {
                doConv(currentInput + k * inputSize, inputWidth, inputHeight,
                       currentKernel + k * kernelSize, kernelWidth,
                       outputLayer, outputWidth, outputHeight,
                       xStride, yStride, xPadding, yPadding
                );
            }
        }
    }
    for (int i = 0; i < weightCount; ++i) {
        kernel[i] /= batchSize;
    }
    if (biases) {
        memset(biases, 0, kernelCount * sizeof(FloatType));
        const FloatType *curDelta = delta;
        for (int m = 0; m < batchSize; ++m) {
            for (int i = 0; i < kernelCount; ++i, curDelta += outputSize) {
                FloatType sum = 0;
                for (int j = 0; j < outputSize; ++j) {
                    sum += curDelta[j];
                }
                biases[i] += sum;
            }
        }
        for (int i = 0; i < kernelCount; ++i) {
            biases[i] /= batchSize;
        }
    }
}

void linearLayerBias(FloatType *vs, int dim, const FloatType *biases, int batchSize)
{
    for (int i = 0; i < batchSize; ++i) {
        FloatType *v = vs + i * dim;
        for (int j = 0; j < dim; ++j) {
            v[j] += biases[j];
        }
    }
}

void convLayerBias(FloatType *m, int r, int c, int channel, const FloatType *biases, int batchSize)
{
    int size = r * c;
    for (int i = 0; i < batchSize; ++i) {
        FloatType *currentM = m + i * size * channel;
        for (int j = 0; j < channel; ++j) {
            FloatType *layer = currentM + j * size;
            const FloatType b = biases[j];
            for (int k = 0; k < size; ++k) {
                layer[k] += b;
            }
        }
    }
}

void initConvKernel(FloatType *k, int size, int inputDim)
{
    std::random_device rd;
    std::normal_distribution<FloatType> distribution(0, std::sqrt(static_cast<FloatType>(2) / inputDim));
    for (int i = 0; i < size; ++i) {
        k[i] = distribution(rd);
    }
}

void initLinearWeights(FloatType *w, int outputDim, int inputDim)
{
    std::random_device rd;
    std::normal_distribution<FloatType> distribution(0, std::sqrt(static_cast<FloatType>(2) / inputDim));
    /*随机生成weight*/
    int weightCount = outputDim * inputDim;
    for (int i = 0; i < weightCount; ++i) {
        w[i] = distribution(rd);
    }
}

void getMNISTBatch(FloatType *data, FloatType *labels,
                   const unsigned char *dataBuffer, const unsigned char *labelBuffer,
                   const int *indices, int count
)
{
    for (int i = 0; i < count; ++i) {
        int index = indices[i];
        const unsigned char *curDataBuffer = dataBuffer + 28 * 28 * index;
        FloatType *curData = data + 28 * 28 * i;
        for (int j = 0; j < 28 * 28; ++j) {
            curData[j] = static_cast<FloatType>(curDataBuffer[j]) / 0xff;
        }
    }
    if (labels) {
        memset(labels, 0, 10 * count * sizeof(FloatType));
        for (int i = 0; i < count; ++i) {
            labels[i * 10 + labelBuffer[indices[i]]] = 1;
        }
    }
}

void getCIFAR10Batch(FloatType *data, FloatType *labels,
                     const unsigned char *buffer,
                     const int *indices, int count
)
{
    for (int i = 0; i < count; ++i) {
        int index = indices[i];
        const unsigned char *b = buffer + index * 3073 + 1;
        FloatType *curData = data + 3072 * i;
        for (int j = 0; j < 3072; ++j) {
            curData[j] = static_cast<FloatType>(b[j]) / 0xff;
        }
    }
    if (labels) {
        memset(labels, 0, count * 10 * sizeof(FloatType));
        for (int i = 0; i < count; ++i) {
            int index = indices[i];
            labels[10 * i + buffer[index * 3073]] = 1;
        }
    }
}

void bnOneDivDev(FloatType *out, const FloatType *var, int size)
{
    for (int i = 0; i < size; ++i) {
        out[i] = static_cast<FloatType>(1) / std::sqrt(var[i] + static_cast<FloatType>(1e-4));
    }
}

void linearDropout(FloatType *v, int dim, const int *ids, int dropoutCount, int batchSize)
{
    for (int i = 0; i < batchSize; ++i) {
        FloatType *cv = v + i * dim;
        for (int j = 0; j < dim; ++j) {
            if (ids[j] < dropoutCount) cv[j] = 0;
        }
    }
}

void batchNormalize(FloatType *out, const FloatType *x, const FloatType *avg, const FloatType *oneDivDev, int dim, int batchSize)
{
    for (int i = 0; i < batchSize; ++i) {
        FloatType *curOut = out + i * dim;
        const FloatType *cx = x + i * dim;
        for (int j = 0; j < dim; ++j) {
            curOut[j] = (cx[j] - avg[j]) * oneDivDev[j];
        }
    }
}

void bnTransform(FloatType *out, const FloatType *normOut, const FloatType *gamma, const FloatType *beta, int dim, int batchSize)
{
    for (int i = 0; i < batchSize; ++i) {
        FloatType *curOut = out + i * dim;
        const FloatType *curNormOut = normOut + i * dim;
        for (int j = 0; j < dim; ++j) {
            curOut[j] = gamma[j] * curNormOut[j] + beta[j];
        }
    }
}

void bnXSubAvg(FloatType *out, const FloatType *x, const FloatType *avg, int dim, int batchSize)
{
    for (int i = 0; i < batchSize; ++i) {
        FloatType *curOut = out + i * dim;
        const FloatType *cx = x + i * dim;
        for (int j = 0; j < dim; ++j) {
            curOut[j] = cx[j] - avg[j];
        }
    }
}

void bnVariance(FloatType *out, const FloatType *xSubAvg, int dim, int batchSize)
{
    for (int i = 0; i < batchSize; ++i) {
        const FloatType *curXSubAvg = xSubAvg + i * dim;
        for (int j = 0; j < dim; ++j) {
            out[j] += curXSubAvg[j] * curXSubAvg[j];
        }
    }
    for (int i = 0; i < dim; ++i) {
        out[i] /= batchSize;
    }
}

void bnDeltaMulCenter(FloatType *out, const FloatType *delta, const FloatType *xSubAvg, int dim, int batchSize)
{
    memset(out, 0, dim * sizeof(FloatType));
    for (int i = 0; i < batchSize; ++i) {
        const FloatType *curDelta = delta + i * dim;
        const FloatType *curXSubAvg = xSubAvg + i * dim;
        for (int j = 0; j < dim; ++j) {
            out[j] += curDelta[j] * curXSubAvg[j];
        }
    }
}

void bnBackProp(FloatType *out, const FloatType *gamma, const FloatType *normDelta, const FloatType *normOut, const FloatType *var, const FloatType *deltaMulCenter, int dim, int batchSize)
{
    for (int i = 0; i < batchSize; ++i) {
        FloatType *curOut = out + i * dim;
        const FloatType *curNormDelta = normDelta + i * dim;
        const FloatType *curNormOut = normOut + i * dim;
        for (int j = 0; j < dim; ++j) {
            curOut[j] = gamma[j] * (curNormDelta[j] - curNormOut[j] * deltaMulCenter[j] / ((var[j] + static_cast<FloatType>(1e-4)) * batchSize));
        }
    }
}

void bnGradients(FloatType *gamma, FloatType *beta, const FloatType *delta, const FloatType *normOut, int dim, int batchSize)
{
    memset(gamma, 0, dim * sizeof(FloatType));
    memset(beta, 0, dim * sizeof(FloatType));
    for (int i = 0; i < batchSize; ++i) {
        const FloatType *curDelta = delta + i * dim;
        const FloatType *curNormOut = normOut + i * dim;
        for (int j = 0; j < dim; ++j) {
            gamma[j] += curDelta[j] * curNormOut[j];
            beta[j] += curDelta[j];
        }
    }
}

void bnGlobalValues(FloatType *globalAvg, FloatType *globalVar, FloatType *globalOneDivDev, const FloatType *avg, const FloatType *var, int dim)
{
    for (int i = 0; i < dim; ++i) {
        globalAvg[i] = globalAvg[i] * static_cast<FloatType>(0.9) + avg[i] * static_cast<FloatType>(0.1);
    }
    for (int i = 0; i < dim; ++i) {
        globalVar[i] = globalVar[i] * static_cast<FloatType>(0.9) + var[i] * static_cast<FloatType>(0.1);
    }
    for (int i = 0; i < dim; ++i) {
        globalOneDivDev[i] = static_cast<FloatType>(1) / std::sqrt(globalVar[i] + static_cast<FloatType>(1e-4));
    }
}

void incArray(int *arr, int len, int bias)
{
    for (int i = 0; i < len; ++i) {
        arr[i] = i + bias;
    }
}

#endif //!ENABLE_CUDA