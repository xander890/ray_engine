#pragma once
#include <memory>
#include <host_device_common.h>
#include "bssrdf_common.h"
#include "mesh_host.h"

class NeuralNetworkSampler
{
public:
	NeuralNetworkSampler(optix::Context & ctx);
	bool on_draw();
	void load(const optix::float3& relative_ior, const ScatteringMaterialProperties & props);
protected:
	optix::Context mContext;

        // Hypernetwork buffers
        std::vector<optix::Buffer> mHyperNetworkWeights;
        std::vector<optix::Buffer> mHyperNetworkBiases;
};

