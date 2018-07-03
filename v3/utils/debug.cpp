/**
 * Created by wjy50 on 18-6-27.
 */

#include <iostream>
#include "debug.h"

NOut::NOut(const char *file)
{
    outputFile = new ofstream(file);

    debugLevel = 0;
}

NOut& NOut::operator<<(const char *__s)
{
    if (debugLevel < DEBUG_LEVEL_THRESHOLD)
        return *this;
    *outputFile << __s;
    cout << __s;
    return *this;
}

NOut& NOut::operator<<(int __i)
{
    if (debugLevel < DEBUG_LEVEL_THRESHOLD)
        return *this;
    *outputFile << __i;
    cout << __i;
    return *this;
}

NOut& NOut::operator<<(long __l)
{
    if (debugLevel < DEBUG_LEVEL_THRESHOLD)
        return *this;
    *outputFile << __l;
    cout << __l;
    return *this;
}

NOut& NOut::operator<<(float __f)
{
    if (debugLevel < DEBUG_LEVEL_THRESHOLD)
        return *this;
    *outputFile << __f;
    cout << __f;
    return *this;
}

NOut& NOut::operator<<(short __s)
{
    if (debugLevel < DEBUG_LEVEL_THRESHOLD)
        return *this;
    *outputFile << __s;
    cout << __s;
    return *this;
}

NOut& NOut::operator<<(double __d)
{
    if (debugLevel < DEBUG_LEVEL_THRESHOLD)
        return *this;
    *outputFile << __d;
    cout << __d;
    return *this;
}

NOut& NOut::operator<<(long long __ll)
{
    if (debugLevel < DEBUG_LEVEL_THRESHOLD)
        return *this;
    *outputFile << __ll;
    cout << __ll;
    return *this;
}

NOut& NOut::operator<<(char __c)
{
    if (debugLevel < DEBUG_LEVEL_THRESHOLD)
        return *this;
    *outputFile << __c;
    if (__c == endl) outputFile->flush();
    cout << __c;
    return *this;
}

void NOut::setDebugLevel(int level)
{
    debugLevel = level;
}

NOut::~NOut()
{
    outputFile->close();
    delete outputFile;
}

bool initialized = false;

unique_ptr<NOut> nOut;

NOut& nout(int debugLevel)
{
    if (!initialized) {
        nOut = make_unique<NOut>();
        initialized = true;
    }
    nOut->setDebugLevel(debugLevel);
    return *(nOut.get());
}