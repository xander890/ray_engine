#pragma once
#include "reference_bssrdf.h"
#include "bssrdf_creator.h"

class ReferenceBSSRDFGPU : public BSSRDFHemisphereSimulated
{
public:
	ReferenceBSSRDFGPU(optix::Context & ctx, const optix::uint2 & hemisphere = optix::make_uint2(160, 40), const unsigned int samples = (int)1e5) : BSSRDFHemisphereSimulated(ctx, hemisphere,samples)
	{
	}

	void init() override;
	void render() override;
	void load_data() override;
	void set_samples(int samples) override;
	bool on_draw(bool show_material_params) override;

	size_t get_samples() override;
	
	optix::Buffer mAtomicPhotonCounterBuffer = nullptr;
	optix::Buffer mPhotonBuffer = nullptr;
	void reset() override;
	unsigned int mBatchIterations = (int)1e4;
	unsigned int mMaxFrames = 1000;
	unsigned long long mPhotons = 0;
};

