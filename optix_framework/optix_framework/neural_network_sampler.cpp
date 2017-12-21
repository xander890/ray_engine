#include "neural_network_sampler.h"
#include "logger.h"

bool NeuralNetworkSampler::on_draw()
{
    return false; // Nothing has changed.
}

void NeuralNetworkSampler::load(float relative_ior, const ScatteringMaterialProperties & props)
{
    //Logger::info << "Log something..." << std::endl;
    // It will be called the next frame, if something changes.
    // You can trigger this by returning true in the on_draw method. 
    // Here you can do context->set stuff, etc.
}
