#include "approximate_dipoles_host.h"
#include "optix_utils.h"
#include "parameter_parser.h"
#include "immediate_gui.h"
#include "scattering_material.h"

ApproximateDipole::ApproximateDipole(optix::Context & ctx, ScatteringDipole::Type type):  BSSRDF(ctx, type)
{
	mProperties.approx_property_A = ConfigParameters::get_parameter<optix::float3>("bssrdf", "approximate_A", optix::make_float3(1), "Approximate value A for approximate dipoles.");
	mProperties.approx_property_s = ConfigParameters::get_parameter<optix::float3>("bssrdf", "approximate_s", optix::make_float3(1), "Approximate value s for approximate dipoles.");
	mSamplingType = SamplingMfpType::to_enum(ConfigParameters::get_parameter<std::string>("bssrdf", "bssrdf_sampling_mfp", SamplingMfpType::to_string(mSamplingType), (std::string("Part of transport/s coeff. used for sampling. Values: ") + ScatteringDipole::get_full_string()).c_str()));
	mProperties.sampling_mfp_s = computeSamplingMfp(mSamplingType, mProperties.approx_property_s);

	mContext["approx_std_bssrdf_props"]->setUserData(sizeof(ApproximateBSSRDFProperties), &mProperties);
}


void ApproximateDipole::load(const ScatteringMaterialProperties & properties)
{
	mContext["approx_std_bssrdf_props"]->setUserData(sizeof(ApproximateBSSRDFProperties), &mProperties);
}

void ApproximateDipole::on_draw()
{
	bool changed = ImmediateGUIDraw::InputFloat3("A", (float*)&mProperties.approx_property_A);
	changed |= ImmediateGUIDraw::InputFloat3("s", (float*)&mProperties.approx_property_s);
	if (changed)
	{
		mProperties.sampling_mfp_s = computeSamplingMfp(mSamplingType, mProperties.approx_property_s);
	}
}
