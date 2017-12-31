#pragma once
#include <device_common_data.h>
#include <material_device.h>
#include <bssrdf_properties.h>
#include <scattering_properties.h>
#include <directional_dipole.h>
#include <standard_dipole.h>
#include <approximate_directional_dipole.h>
#include <approximate_standard_dipole.h>
#include <bssrdf_properties.h>
#include <quantized_diffusion.h>
#include <photon_beam_diffusion.h>
#include <empirical_bssrdf_device.h>
//#define INCLUDE_PROGRAMS_ONLY
#include <forward_dipole.h>

using optix::float3;
rtDeclareVariable(ScatteringDipole::Type, selected_bssrdf, , );

__forceinline__ __device__ float3 bssrdf(const BSSRDFGeometry & geometry, const float recip_ior,
	const MaterialDataCommon& material)
{   
	switch (selected_bssrdf)
	{
	case ScatteringDipole::APPROX_DIRECTIONAL_DIPOLE_BSSRDF:
		return approximate_directional_dipole_bssrdf(geometry, recip_ior, material);
	case ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF:
		return directional_dipole_bssrdf(geometry, recip_ior, material);
	case ScatteringDipole::APPROX_STANDARD_DIPOLE_BSSRDF:
		return approximate_standard_dipole_bssrdf(geometry, recip_ior, material);
	case ScatteringDipole::QUANTIZED_DIFFUSION_BSSRDF:
		return quantized_diffusion_bssrdf(geometry, recip_ior, material);
	case ScatteringDipole::PHOTON_BEAM_DIFFUSION_BSSRDF:
		return photon_beam_diffusion_bssrdf(geometry, recip_ior, material);
	case ScatteringDipole::FORWARD_SCATTERING_DIPOLE_BSSRDF:
		return forward_dipole_bssrdf(geometry, recip_ior, material);
    case ScatteringDipole::EMPIRICAL_BSSRDF:
        return eval_empbssrdf(geometry, recip_ior, material);
	case ScatteringDipole::STANDARD_DIPOLE_BSSRDF:
	default:
		return standard_dipole_bssrdf(geometry, recip_ior, material);
	}
}

__forceinline__ __device__ float3 get_beam_transmittance(const float depth, const ScatteringMaterialProperties& properties)
{
	switch (selected_bssrdf)
	{
	case ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF:
	case ScatteringDipole::APPROX_DIRECTIONAL_DIPOLE_BSSRDF:
		return exp(-depth*properties.deltaEddExtinction);
	case ScatteringDipole::APPROX_STANDARD_DIPOLE_BSSRDF:
    case ScatteringDipole::EMPIRICAL_BSSRDF:
	case ScatteringDipole::STANDARD_DIPOLE_BSSRDF:
	case ScatteringDipole::QUANTIZED_DIFFUSION_BSSRDF:
	case ScatteringDipole::PHOTON_BEAM_DIFFUSION_BSSRDF:
	default:
		return exp(-depth*properties.extinction);
	}
}

__forceinline__ __device__ float get_sampling_mfp(const ScatteringMaterialProperties& properties)
{
	switch (selected_bssrdf)
	{
	case ScatteringDipole::APPROX_DIRECTIONAL_DIPOLE_BSSRDF:
	case ScatteringDipole::APPROX_STANDARD_DIPOLE_BSSRDF:
		return approx_std_bssrdf_props.sampling_mfp_s;
    case ScatteringDipole::EMPIRICAL_BSSRDF:
	case ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF:
	case ScatteringDipole::STANDARD_DIPOLE_BSSRDF:
	case ScatteringDipole::QUANTIZED_DIFFUSION_BSSRDF:
	case ScatteringDipole::PHOTON_BEAM_DIFFUSION_BSSRDF:
	default:
		return properties.sampling_mfp_tr;
	}
}
