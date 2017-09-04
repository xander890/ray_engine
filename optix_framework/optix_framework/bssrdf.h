#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>
#include <directional_dipole.h>
#include <standard_dipole.h>

using optix::float3;

__forceinline__ __device__ float3 bssrdf(const float3& _xi, const float3& _ni, const float3& _w12,
	const float3& _xo, const float3& _no,
	const ScatteringMaterialProperties& properties)
{
	switch (properties.selected_bssrdf)
	{
	case DIRECTIONAL_DIPOLE_BSSRDF:
		return directional_dipole_bssrdf(_xi, _ni, _w12, _xo, _no, properties);
	case STANDARD_DIPOLE_BSSRDF:
	default:
		return standard_dipole_bssrdf(length(_xo - _xi), properties);
	}
}

__forceinline__ __device__ float3 get_beam_transmittance(const float depth, const ScatteringMaterialProperties& properties)
{
	switch (properties.selected_bssrdf)
	{
	case DIRECTIONAL_DIPOLE_BSSRDF:
		return exp(-depth*properties.deltaEddExtinction);
	case STANDARD_DIPOLE_BSSRDF:
	default:
		return exp(-depth*properties.extinction);
	}
}