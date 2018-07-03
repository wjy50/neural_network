/**
 * Created by wjy50 on 18-6-15.
 */

#ifndef NEURAL_NETWORK_RESIDUALBLOCK_H
#define NEURAL_NETWORK_RESIDUALBLOCK_H


#include <vector>
#include "base/LayerBase.h"

class ResidualBlock : public LayerBase
{
private:
    std::vector<LayerBase*> layers;

    bool built;
protected:
    bool needIndependentOutput() override;

    void computeGradients() override;
public:
    explicit ResidualBlock(int dim);

    void addLayer(LayerBase *layer);

    const FloatType *feedForward(const FloatType *x, int count) override;

    const FloatType *feedForwardForOptimization(const FloatType *x) override;

    void backPropagate(const FloatType *y) override;

    void updateParameters() override;

    bool needBackPropAtFirst() override;

    FloatType *getDelta() override;
};


#endif //NEURAL_NETWORK_RESIDUALBLOCK_H
