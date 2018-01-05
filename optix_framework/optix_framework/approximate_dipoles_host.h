#pragma once
#include "bssrdf_host.h"

class ApproximateDipole : public BSSRDF
{
public:
	ApproximateDipole(optix::Context & ctx, ScatteringDipole::Type type);
	void load(const float relative_ior, const ScatteringMaterialProperties &props) override;
	void on_draw() override;
private:
	ApproximateBSSRDFProperties mProperties;
	SamplingMfpType::Type mSamplingType;
};
