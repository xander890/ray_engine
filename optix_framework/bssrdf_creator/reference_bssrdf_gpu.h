#pragma once
#include "reference_bssrdf.h"
#include "bssrdf_creator.h"

class ReferenceBSSRDFGPU : public BSSRDFRendererSimulated
{
public:
	ReferenceBSSRDFGPU(optix::Context & ctx, const OutputShape::Type shape = OutputShape::HEMISPHERE, const optix::int2 & shape_size = optix::make_int2(-1), const unsigned int samples = (int)1e8) : BSSRDFRendererSimulated(ctx, shape, shape_size, samples)
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
	unsigned int mMaxFrames = 100000;
	unsigned long long mPhotons = 0;
};

