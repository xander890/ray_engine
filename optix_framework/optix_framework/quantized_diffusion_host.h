#pragma once
#include "bssrdf_host.h"

class QuantizedDiffusion : public BSSRDF
{
public:
	QuantizedDiffusion(optix::Context & ctx);
	void load(const optix::float3 &relative_ior, const ScatteringMaterialProperties &props) override;
	bool on_draw() override;

private:
	QuantizedDiffusionProperties mProperties;
	optix::Buffer mPropertyBuffer;
	optix::Buffer mBSSRDFPrecomputed;
	bool mHasChanged = true;

	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(
			cereal::base_class<BSSRDF>(this),
			cereal::make_nvp("maximum_distance", mProperties.max_dist_bssrdf),
			cereal::make_nvp("precomputed_size", mProperties.precomputed_bssrdf_size),
			cereal::make_nvp("use_precomputed", mProperties.use_precomputed_qd)
		);
	}
};
