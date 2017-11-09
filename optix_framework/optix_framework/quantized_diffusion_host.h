#pragma once
#include "bssrdf_host.h"

class QuantizedDiffusion : public BSSRDF
{
public:
	QuantizedDiffusion(optix::Context & ctx);
	void load(const ScatteringMaterialProperties & props) override;
	void on_draw() override;

private:
	QuantizedDiffusionProperties mProperties;
	optix::Buffer mPropertyBuffer;
	optix::Buffer mBSSRDFPrecomputed;
	bool mHasChanged = true;
};
