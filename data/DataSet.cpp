/**
 * Created by wjy50 on 18-5-2.
 */

#include "DataSet.h"

DataSet::DataSet()
{
    normalizer = nullptr;
}

void DataSet::setNormalizer(DataNormalizer *normalizer)
{
    this->normalizer = normalizer;
}