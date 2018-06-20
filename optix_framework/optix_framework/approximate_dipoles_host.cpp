#include "approximate_dipoles_host.h"
#include "optix_host_utils.h"
#include "immediate_gui.h"
#include "scattering_material.h"

ApproximateDipole::ApproximateDipole(optix::Context & ctx, ScatteringDipole::Type type):  BSSRDF(ctx, type)
{
	mContext["approx_std_bssrdf_props"]->setUserData(sizeof(ApproximateBSSRDFProperties), &mProperties);
}


void ApproximateDipole::load(const optix::float3 &relative_ior, const ScatteringMaterialProperties &properties)
{
	mContext["approx_std_bssrdf_props"]->setUserData(sizeof(ApproximateBSSRDFProperties), &mProperties);
}

bool ApproximateDipole::on_draw()
{
	bool changed = ImmediateGUIDraw::InputFloat3("A", (float*)&mProperties.approx_property_A);
	changed |= ImmediateGUIDraw::InputFloat3("s", (float*)&mProperties.approx_property_s);
	return changed;
}
