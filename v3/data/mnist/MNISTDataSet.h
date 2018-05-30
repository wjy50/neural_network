/**
 * Created by wjy50 on 18-5-18.
 */

#ifndef NEURAL_NETWORK_MNISTDATASET_H
#define NEURAL_NETWORK_MNISTDATASET_H


#include "../base/DataSetBase.h"
#include "../../utils/UniquePointerExt.h"

class MNISTDataSet : public DataSetBase
{
private:
    int count;

    std::unique_ptr<unsigned char[]> buffer;
    std::unique_ptr<unsigned char[]> labelBuffer;
public:
    MNISTDataSet(const char *imagePath, const char *labelPath);

    int getCount() override;

    void getBatch(FloatType *data, FloatType *labels, const int *indices, int count) override;
};


#endif //NEURAL_NETWORK_MNISTDATASET_H
