#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>
#include <bssrdf_properties.h>
#include <directional_dipole.h>
#include <standard_dipole.h>
#include <approximate_directional_dipole.h>
#include <approximate_standard_dipole.h>
#include <bssrdf_properties.h>
#include <quantized_diffusion.h>
#include <photon_beam_diffusion.h>
#include <forward_dipole.h>

using optix::float3;
rtDeclareVariable(ScatteringDipole::Type, selected_bssrdf, , );

__forceinline__ __device__ float3 bssrdf(const float3& _xi, const float3& _ni, const float3& _w12,
	const float3& _xo, const float3& _no, const float3 & _w21,
	const ScatteringMaterialProperties& properties)
{
	switch (selected_bssrdf)
	{
	case ScatteringDipole::APPROX_DIRECTIONAL_DIPOLE_BSSRDF:
		return approximate_directional_dipole_bssrdf(_xi, _ni, _w12, _xo, _no, _w21, properties);
	case ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF:
		return directional_dipole_bssrdf(_xi, _ni, _w12, _xo, _no, properties);
	case ScatteringDipole::APPROX_STANDARD_DIPOLE_BSSRDF:
		return approximate_standard_dipole_bssrdf(length(_xo - _xi), properties);
	case ScatteringDipole::QUANTIZED_DIFFUSION_BSSRDF:
		return quantized_diffusion_bssrdf(_xi, _ni, _w12, _xo, _no, properties);
	case ScatteringDipole::PHOTON_BEAM_DIFFUSION_BSSRDF:
		return photon_beam_diffusion_bssrdf(_xi, _ni, _w12, _xo, _no, properties);
	case ScatteringDipole::FORWARD_SCATTERING_DIPOLE_BSSRDF:
		return forward_dipole_bssrdf(_xi, _ni, _w12, _xo, _no, _w21, properties);
	case ScatteringDipole::STANDARD_DIPOLE_BSSRDF:
	default:
		return standard_dipole_bssrdf(length(_xo - _xi), properties);
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
	case ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF:
	case ScatteringDipole::STANDARD_DIPOLE_BSSRDF:
	case ScatteringDipole::QUANTIZED_DIFFUSION_BSSRDF:
	case ScatteringDipole::PHOTON_BEAM_DIFFUSION_BSSRDF:
	default:
		return properties.sampling_mfp_tr;
	}
}