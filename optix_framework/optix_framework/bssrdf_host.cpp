#include "bssrdf_host.h"
#include "immediate_gui.h"
#include "forward_dipole_host.h"
#include "quantized_diffusion_host.h"
#include "approximate_dipoles_host.h"
#include "string_utils.h"
#include "empirical_bssrdf_host.h"

BSSRDF::BSSRDF(optix::Context & ctx, ScatteringDipole::Type type)
{
	mType = type;
	mContext = ctx;
}

bool BSSRDF::on_draw()
{
	return false;
}

void BSSRDF::load(const optix::float3 &, const ScatteringMaterialProperties &)
{
}

optix::float3 BSSRDF::get_sampling_inverse_mean_free_path(const ScatteringMaterialProperties &props)
{ 
	return props.transport; 
}

ScatteringDipole::Type BSSRDF::get_type() 
{ 
	return mType; 
}

std::unique_ptr<BSSRDF> BSSRDF::create(optix::Context & ctx, ScatteringDipole::Type type)
{
	switch (type)
	{
	case ScatteringDipole::APPROX_DIRECTIONAL_DIPOLE_BSSRDF:
		return std::unique_ptr<BSSRDF>(new ApproximateDipole(ctx, ScatteringDipole::APPROX_DIRECTIONAL_DIPOLE_BSSRDF));
	case ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF:
		return std::unique_ptr<BSSRDF>(new BSSRDF(ctx, ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF));
	case ScatteringDipole::APPROX_STANDARD_DIPOLE_BSSRDF:
		return std::unique_ptr<BSSRDF>(new ApproximateDipole(ctx, ScatteringDipole::APPROX_STANDARD_DIPOLE_BSSRDF));
	case ScatteringDipole::QUANTIZED_DIFFUSION_BSSRDF:
		return std::unique_ptr<BSSRDF>(new QuantizedDiffusion(ctx));
	case ScatteringDipole::PHOTON_BEAM_DIFFUSION_BSSRDF:
		return std::unique_ptr<BSSRDF>(new BSSRDF(ctx, ScatteringDipole::PHOTON_BEAM_DIFFUSION_BSSRDF));
	case ScatteringDipole::FORWARD_SCATTERING_DIPOLE_BSSRDF:
		return std::unique_ptr<BSSRDF>(new ForwardDipole(ctx));
	case ScatteringDipole::EMPIRICAL_BSSRDF:
		return std::unique_ptr<BSSRDF>(new EmpiricalBSSRDF(ctx));
	case ScatteringDipole::STANDARD_DIPOLE_BSSRDF:
	default:
		return std::unique_ptr<BSSRDF>(new BSSRDF(ctx, ScatteringDipole::STANDARD_DIPOLE_BSSRDF));
	}
}

bool BSSRDF::bssrdf_selection_gui(ScatteringDipole::Type & type, std::string id)
{
	std::string dipoles = "";
	ScatteringDipole::Type t = ScatteringDipole::first();
	do
	{
		dipoles += prettify(ScatteringDipole::to_string(t)) + '\0';
		t = ScatteringDipole::next(t);
	} while (t != ScatteringDipole::NotValidEnumItem);

#define ID_STRING(x,id) (std::string(x) + "##" + id + x).c_str()
	if (ImmediateGUIDraw::Combo(ID_STRING("Dipole", id), (int*)&type, dipoles.c_str(), ScatteringDipole::count()))
	{
		return true;
	}
	return false;
#undef ID_STRING
}

