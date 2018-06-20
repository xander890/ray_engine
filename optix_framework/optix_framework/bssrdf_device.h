#pragma once
#include <device_common.h>
#include <bssrdf_common.h>
#include <scattering_properties.h>

// Dipoles
#ifdef FORWARD_DIPOLE_ONLY
#include <forward_dipole_device.h>
#else
#include <approximate_dipoles_device.h>
#include <directional_dipole_device.h>
#include <standard_dipole_device.h>
#include <quantized_diffusion_device.h>
#include <photon_beam_diffusion_device.h>
#include <empirical_bssrdf_device.h>
#endif

// The type of BSSRDF to render
rtDeclareVariable(ScatteringDipole::Type, selected_bssrdf, , );

/*
 *  Calculates a BSSRDF contribution given geometry, material and sampler.
 *  recip_ior is the reciprocal of the interface IOR already accounting for the normal.
 */
_fn optix::float3 bssrdf(const BSSRDFGeometry & geometry, const float recip_ior,
	const MaterialDataCommon& material, BSSRDFFlags::Type flags, TEASampler & sampler)
{   
#ifdef FORWARD_DIPOLE_ONLY
    return forward_dipole_bssrdf(geometry, recip_ior, material, flags, sampler);
#else
    optix::float3 S = optix::make_float3(0.0f);
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

_fn optix::float3 get_beam_transmittance(const float depth, const ScatteringMaterialProperties& properties)
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

