#pragma once
#include <optix_world.h>
#include "host_device_common.h"
#include "optical_helper.h"

struct ScatteringMaterialProperties
{
	// base parameters
	optix::float3 absorption					DEFAULT(optix::make_float3(1));
    optix::float3 scattering					DEFAULT(optix::make_float3(0));
    optix::float3 meancosine					DEFAULT(optix::make_float3(0));

    // derived parameters, no need to initialize
    optix::float3 extinction;
    optix::float3 reducedExtinction;
    optix::float3 D;
    optix::float3 albedo;
    optix::float3 transport;
    float C_phi;
    optix::float3 reducedAlbedo;
    float C_phi_inv; // C_phi(1 / eta)
    optix::float3 de;
    float C_E;
    optix::float3 three_D;
    float A;
    optix::float3 rev_D;
    float global_coeff;
    optix::float3 two_a_de;
	int pad;
    optix::float3 one_over_three_ext;
	float C_phi_norm; // 1 / (4 * C_phi(1 / eta)) = 1 / (1 - 2 * C1(1/eta)) 
	optix::float3 deltaEddExtinction;
};

_fn void fill_scattering_parameters(ScatteringMaterialProperties & properties, const float scale, const float ior, const optix::float3 & absorption, const optix::float3 & scattering, const optix::float3 & asymmetry)
{
	const float inverse_relative_ior = 1.0f / ior;
	properties.absorption = max(absorption, optix::make_float3(1.0e-8f)) * scale;
	properties.scattering = scattering * scale;
	properties.meancosine = asymmetry;
	properties.deltaEddExtinction = properties.scattering*(1.0f - properties.meancosine*properties.meancosine) + properties.absorption;

	auto reducedScattering = properties.scattering * (optix::make_float3(1.0f) - properties.meancosine);
	properties.reducedExtinction = reducedScattering + properties.absorption;
	properties.D = optix::make_float3(1.0f) / (3.f * properties.reducedExtinction);
	properties.transport = sqrt(3 * properties.absorption*properties.reducedExtinction);
	properties.C_phi = C_phi(ior);
	properties.C_phi_inv = C_phi(inverse_relative_ior);
	properties.C_phi_norm = 1.0f/(1.0f - 2.0f * C_1(inverse_relative_ior));
	properties.C_E = C_E(ior);
	properties.reducedAlbedo = reducedScattering / properties.reducedExtinction;
	properties.de = 2.131f * properties.D / sqrt(properties.reducedAlbedo);
	properties.A = (1.0f - properties.C_E) / (2.0f * properties.C_phi);
	properties.extinction = properties.scattering + properties.absorption;
	properties.three_D = 3 * properties.D;
	properties.rev_D = (3.f * properties.reducedExtinction);
	properties.two_a_de = 2.0f * properties.A * properties.de;
	properties.global_coeff = 1.0f / (4.0f * properties.C_phi_inv) * 1.0f / (4.0f * M_PIf * M_PIf);
	properties.one_over_three_ext = optix::make_float3(1.0) / (3.0f * properties.extinction);
	properties.albedo = properties.scattering / properties.extinction;
}

_fn void fill_scattering_parameters_alternative(ScatteringMaterialProperties & properties, const float scale, const float ior, const optix::float3 & albedo, const optix::float3 & extinction, const optix::float3 & asymmetry)
{
	optix::float3 scattering = albedo*extinction;
	optix::float3 absorption = extinction - scattering;
	fill_scattering_parameters(properties, scale, ior, absorption, scattering, asymmetry);
}