#pragma once
#include <memory>
#include <host_device_common.h>
#include "bssrdf_properties.h"
#include "mesh.h"

class NeuralNetworkSampler
{
public:
	NeuralNetworkSampler(optix::Context & ctx) : mContext(ctx) {}
	bool on_draw();
	void load(float relative_ior, const ScatteringMaterialProperties & props);
protected:
	optix::Context mContext;
};

