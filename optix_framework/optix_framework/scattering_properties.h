#pragma once
#include <optix_world.h>
#include "host_device_common.h"

namespace ScatteringDipole
{
	enum ScatteringDipoleEnum {
		STANDARD_DIPOLE_BSSRDF = 0,
		DIRECTIONAL_DIPOLE_BSSRDF = 1,
		APPROX_STANDARD_DIPOLE_BSSRDF = 2,
		APPROX_DIRECTIONAL_DIPOLE_BSSRDF = 3,
		BSSRDF_COUNT = 4
	};
};
struct ScatteringMaterialProperties
{
	// base parameters
	optix::float3 absorption				DEFAULT(optix::make_float3(1));
    optix::float3 scattering				DEFAULT(optix::make_float3(0));
    optix::float3 meancosine				DEFAULT(optix::make_float3(0));
	int selected_bssrdf						DEFAULT(ScatteringDipole::APPROX_STANDARD_DIPOLE_BSSRDF);

    // derived parameters, no need to initialize
    optix::float3 extinction;
    optix::float3 reducedExtinction;
    optix::float3 D;
    optix::float3 albedo;
    optix::float3 transport;
    float C_phi;
    optix::float3 reducedAlbedo;
    float C_phi_inv;
    optix::float3 de;
    float C_E;
    optix::float3 three_D;
    float A;
    optix::float3 rev_D;
    float global_coeff;
    optix::float3 two_a_de;
    float mean_transport;
    optix::float3 one_over_three_ext;
    float min_transport;
    optix::float3 deltaEddExtinction;
};