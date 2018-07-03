/**
 * Created by wjy50 on 18-6-27.
 */

#ifndef NEURAL_NETWORK_DEBUG_H
#define NEURAL_NETWORK_DEBUG_H

#include <fstream>
#include "UniquePointerExt.h"

#define DEBUG_LEVEL_THRESHOLD 0

#define DEBUG_LEVEL_VERBOSE 0

#define DEBUG_LEVEL_INFO 1

#define DEBUG_LEVEL_DEBUG 2

#define DEBUG_LEVEL_IMPORTANT 3

#define endl '\n'

using namespace std;

class NOut
{
private:
    ofstream *outputFile;

    int debugLevel;
public:
    explicit NOut(const char *file = "nn.log");

    NOut&operator<<(const char* __s);

    NOut&operator<<(int __i);

    NOut&operator<<(long __l);

    NOut&operator<<(long long __ll);

    NOut&operator<<(short __s);

    NOut&operator<<(float __f);

    NOut&operator<<(double __d);

    NOut&operator<<(char __c);

    void setDebugLevel(int level);

    ~NOut();
};

extern unique_ptr<NOut> nOut;

extern bool initialized;

NOut& nout(int level = 0);

#endif //NEURAL_NETWORK_DEBUG_H
