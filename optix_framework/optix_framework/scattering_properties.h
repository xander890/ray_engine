#pragma once
#include<optix_world.h>

#define STANDARD_DIPOLE_BSSRDF 0
#define DIRECTIONAL_DIPOLE_BSSRDF 1
#define APPROX_STANDARD_DIPOLE_BSSRDF 2
#define APPROX_DIRECTIONAL_DIPOLE_BSSRDF 3
#define BSSRDF_COUNT 4

struct ScatteringMaterialProperties
{
    optix::float3 absorption;
    optix::float3 scattering;
    optix::float3 meancosine;

    // derived parameters
    optix::float3 extinction;
    optix::float3 reducedExtinction;
    optix::float3 D;
    optix::float3 albedo;
    // base parameters
    float relative_ior; //The usual assumption is that this can be intercheangeably the material ior or the ratio between it and air (ior = 1)
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
	int selected_bssrdf;
};