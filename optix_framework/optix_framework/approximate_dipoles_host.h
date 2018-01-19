#pragma once
#include "bssrdf_host.h"

class ApproximateDipole : public BSSRDF
{
public:
	ApproximateDipole(optix::Context & ctx, ScatteringDipole::Type type);
	void load(const float relative_ior, const ScatteringMaterialProperties &props) override;
	bool on_draw() override;
    optix::float3 get_sampling_inverse_mean_free_path(const ScatteringMaterialProperties &props) override { return mProperties.approx_property_s; }

private:
	ApproximateBSSRDFProperties mProperties;
};
