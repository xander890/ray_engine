#pragma once
#include <optix_world.h>
#include "host_device_common.h"

#define IMPROVED_ENUM_NAME ScatteringDipole
#define IMPROVED_ENUM_LIST	ENUMITEM_VALUE(STANDARD_DIPOLE_BSSRDF,0) \
							ENUMITEM_VALUE(DIRECTIONAL_DIPOLE_BSSRDF,1) \
							ENUMITEM_VALUE(APPROX_STANDARD_DIPOLE_BSSRDF,2) \
							ENUMITEM_VALUE(APPROX_DIRECTIONAL_DIPOLE_BSSRDF,3) 
#include "improved_enum.def"

struct ScatteringMaterialProperties
{
	// base parameters
	optix::float3 absorption					DEFAULT(optix::make_float3(1));
    optix::float3 scattering					DEFAULT(optix::make_float3(0));
    optix::float3 meancosine					DEFAULT(optix::make_float3(0));
	ScatteringDipole::Type	selected_bssrdf		DEFAULT(ScatteringDipole::APPROX_STANDARD_DIPOLE_BSSRDF);

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
    float sampling_mfp_tr;
    optix::float3 one_over_three_ext;
	int pad;
	optix::float3 deltaEddExtinction;

	optix::float3 approx_property_A		DEFAULT(optix::make_float3(1));
	int pad0;
	optix::float3 approx_property_s		DEFAULT(optix::make_float3(1));
	float sampling_mfp_s;
};