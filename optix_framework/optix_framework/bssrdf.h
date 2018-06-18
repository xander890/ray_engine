#pragma once
#include <device_common_data.h>
#include <material_device.h>
#include <bssrdf_properties.h>
#include <scattering_properties.h>
#include <approximate_directional_dipole.h>
#include <approximate_standard_dipole.h>

#ifdef FORWARD_DIPOLE_ONLY
#include <forward_dipole.h>
#else
#include <directional_dipole.h>
#include <standard_dipole.h>
#include <bssrdf_properties.h>
#include <quantized_diffusion.h>
#include <photon_beam_diffusion.h>
#include <empirical_bssrdf_device.h>
#endif

using optix::float3;
rtDeclareVariable(ScatteringDipole::Type, selected_bssrdf, , );

_fn float3 bssrdf(const BSSRDFGeometry & geometry, const float recip_ior,
	const MaterialDataCommon& material, BSSRDFFlags::Type flags, TEASampler & sampler)
{   
#ifdef FORWARD_DIPOLE_ONLY
    return forward_dipole_bssrdf(geometry, recip_ior, material, flags, sampler);
#else
    float3 S = make_float3(0.0f);
    if(selected_bssrdf == ScatteringDipole::APPROX_DIRECTIONAL_DIPOLE_BSSRDF)
    {
        S = approximate_directional_dipole_bssrdf(geometry, recip_ior, material, flags, sampler);
    }
    else if(selected_bssrdf ==  ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF)
    {
        S = directional_dipole_bssrdf(geometry, recip_ior, material, flags, sampler);
    }
    else if(selected_bssrdf == ScatteringDipole::EMPIRICAL_BSSRDF)
    {
        S = eval_empbssrdf(geometry, recip_ior, material, flags, sampler);
    }
    else if(selected_bssrdf == ScatteringDipole::APPROX_STANDARD_DIPOLE_BSSRDF)
    {
        S = approximate_standard_dipole_bssrdf(geometry, recip_ior, material, flags, sampler);
    }
    else if(selected_bssrdf == ScatteringDipole::QUANTIZED_DIFFUSION_BSSRDF)
    {
        S = quantized_diffusion_bssrdf(geometry, recip_ior, material, flags, sampler);
    }
    else if(selected_bssrdf == ScatteringDipole::PHOTON_BEAM_DIFFUSION_BSSRDF)
    {
        S = photon_beam_diffusion_bssrdf(geometry, recip_ior, material, flags, sampler);
    }
    //if(selected_bssrdf == ScatteringDipole::STANDARD_DIPOLE_BSSRDF)
    else
    {
        S = standard_dipole_bssrdf(geometry, recip_ior, material, flags, sampler);
    }
    return S;
#endif
}

_fn float3 get_beam_transmittance(const float depth, const ScatteringMaterialProperties& properties)
{
	switch (selected_bssrdf)
	{
	case ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF:
	case ScatteringDipole::APPROX_DIRECTIONAL_DIPOLE_BSSRDF:
		return exp(-depth*properties.deltaEddExtinction);
	case ScatteringDipole::FORWARD_SCATTERING_DIPOLE_BSSRDF:
	case ScatteringDipole::APPROX_STANDARD_DIPOLE_BSSRDF:
    case ScatteringDipole::EMPIRICAL_BSSRDF:
	case ScatteringDipole::STANDARD_DIPOLE_BSSRDF:
	case ScatteringDipole::QUANTIZED_DIFFUSION_BSSRDF:
	case ScatteringDipole::PHOTON_BEAM_DIFFUSION_BSSRDF:
	default:
		return exp(-depth*properties.extinction);
	}
}

