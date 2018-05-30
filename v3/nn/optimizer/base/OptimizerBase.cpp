/**
 * Created by wjy50 on 18-5-18.
 */

#include <cassert>
#include "OptimizerBase.h"

OptimizerBase::OptimizerBase()
{
    paramCount = 0;
    params = nullptr;
    gradients = nullptr;
    attached = false;
}

void OptimizerBase::attachToLayer(int paramCount, FloatType *params, FloatType *gradients)
{
    assert(!attached);
    this->paramCount = paramCount;
    this->params = params;
    this->gradients = gradients;
    attached = true;
    onAttachedToLayer();
}

void OptimizerBase::onAttachedToLayer() {}