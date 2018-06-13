#pragma once
#include "bssrdf_host.h"
#include "optix_serialize.h"

class ApproximateDipole : public BSSRDF
{
public:
	ApproximateDipole(optix::Context & ctx, ScatteringDipole::Type type);
	void load(const optix::float3 &relative_ior, const ScatteringMaterialProperties &props) override;
	bool on_draw() override;
    optix::float3 get_sampling_inverse_mean_free_path(const ScatteringMaterialProperties &props) override { return mProperties.approx_property_s; }

private:
	ApproximateBSSRDFProperties mProperties;

	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		archive( cereal::base_class<BSSRDF>(this), cereal::make_nvp("A", mProperties.approx_property_A), cereal::make_nvp("s", mProperties.approx_property_s));
	}
};

