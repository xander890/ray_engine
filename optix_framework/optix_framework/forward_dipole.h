#pragma once
#include "host_device_common.h"
#include "bssrdf_properties.h"
#include "scattering_properties.h"
#include <material.h>
#ifdef __CUDACC__
#include <material_device.h>
#endif

#ifdef INCLUDE_PROGRAMS_ONLY
#include "forward_dipole_defines.h"

rtDeclareVariable(rtCallableProgramId<Float(const ForwardDipoleMaterial, Float3, Float3, Float3, Float)>, evalMonopole, , );
rtDeclareVariable(rtCallableProgramId<Float(const ForwardDipoleMaterial, const ForwardDipoleProperties, Float3, Float3, Float3, const Float3*, Float3, Float&, unsigned int &)>, sampleLengthDipole, , );
rtDeclareVariable(rtCallableProgramId<Float(const ForwardDipoleMaterial, const ForwardDipoleProperties, Float3, Float3, Float3, Float3, Float3, Float)>, evalDipole, , );

#else
#include "forward_dipole_sampling.h"
#include "optix_helpers.h"
#include "sampler.h"

__device__ __host__ __forceinline__ Float evalMonopole(
	const ForwardDipoleMaterial material,
	Float3 u0, Float3 uL, Float3 R, Float length) {
	FSAssert(abs(optix::length(u0) - 1) < 1e-6);
	FSAssert(abs(optix::length(uL) - 1) < 1e-6);

	double C, D, E, F;
	calcValues(length, material.sigma_s, material.sigma_a, material.mu, C, D, E, F);

	/* We regularized the sampling of u0, so we should be consistent here.
	* NOTE: but E can still blow up in the final expression for G (TODO
	* does this happen?) */
	// DOUBLE
	Float3 H = E * R - D*uL;
	double lHl = optix::length(H);
	Float3 Hnorm = Float3(H / lHl);
	Float lHlreg = (lHl > 1. / MTS_FWDSCAT_DIRECTION_MIN_MU) ?
		1. / MTS_FWDSCAT_DIRECTION_MIN_MU : lHl;
	Float cosTheta = clamp(dot(u0, Hnorm), -1., 1.);

	double N = absorptionAndFloat3izationConstant(length, material.sigma_s, material.sigma_a, material.mu);
	double G = N * exp(-C + E*dot(R, uL) + lHlreg*cosTheta - F*dot(R, R));
	//Non-regularized:
	//double G = N * exp(-C - D*dot(u0,uL) + E*(dot(R,u0) + dot(R,uL)) - F*R.lengthSquared());

	// Note: fastmath compiler flags may change the order of the operations...
	/* We only care for cancellations if the result is sufficiently large
	* (otherwise exp(epsilon) ~= 1 anyway) */
	if (abs(E*dot(R, uL)) > 1e3)
		CancellationCheck(-C, E*dot(R, uL));
	if (abs(lHlreg*cosTheta) > 1e3)
		CancellationCheck(-C + E*dot(R, uL), (double)lHlreg*cosTheta);
	if (abs(F*dot(R, R)) > 1e3)
		CancellationCheck(-C + E*dot(R, uL) + lHlreg*cosTheta, -F*dot(R, R));

#ifdef MTS_FWDSCAT_DEBUG
	if (!isfinite(G) || G < 0) {
		Log(EWarn, "Invalid G in evalMonopole(): "
			"%e; s %e C %e D %e E %e F %e Rsq %e u0dotuL %e",
			G, length, C, D, E, F, dot(R, R), dot(u0, uL));
		return 0;
	}
#endif
	return G;
}

__device__ __host__ __forceinline__ Float evalPlaneSource(
	const Float sigma_s,
	const Float sigma_a,
	const Float mu,
	Float3 u0, Float3 uL,
	Float3 n, Float Rz, Float length) {

	FSAssert(abs(optix::length(u0) - 1) < 1e-6);
	FSAssert(abs(optix::length(uL) - 1) < 1e-6);

	double C, D, E, F;
	calcValues(length, sigma_s, sigma_a, mu, C, D, E, F);


	Float u0z = dot(u0, n);
	Float uLz = dot(uL, n);

	double result = absorptionAndFloat3izationConstant(length, sigma_s, sigma_a, mu)
		* M_PI / F * exp(
			E*E / 4 / F*(2 + 2 * dot(u0, uL) - (u0z + uLz)*(u0z + uLz))
			- D*dot(u0, uL)
			- C
			+ E*Rz * (u0z + uLz)
			- F*Rz*Rz);

	if (!isfinite(result)) {
		Log(EWarn, "non-finite result %lf", result);
		return 0;
	}
	return result;
}

__device__ __host__ Float evalDipole(
	const ForwardDipoleMaterial material,
	const ForwardDipoleProperties props,
	Float3 n0, Float3 u0,
	Float3 nL, Float3 uL,
	Float3 R, Float length
) {

	/* If reciprocal is requested, nL should be finite and uL_external should point
	* along nL. */
	FSAssert(!props.reciprocal || isfinited(nL));
	FSAssert(!props.reciprocal || dot(uL, nL) >= -Epsilon); // positive with small margin for roundoff errors
													  /* Handle eta != 1 case by 'refracting' the 'external' directions
													  * u0_external and uL_external to 'internal' directions u0 and uL. We
													  * keep the directions pointing along the propagation direction of
													  * light (i.e. not the typical refract as in BSDFs, for instance, which
													  * flips to the other side of the boundary). */
													  //Float _cosThetaT, F0, FL;
													  //Float3 u0 = refract(-u0_external, n0, m_eta, _cosThetaT, F0);
													  //Float3 uL = -refract(uL_external, nL, m_eta, _cosThetaT, FL);
													  //Float fresnelTransmittance = includeFresnelTransmittance? (1 - F0)*(1 - FL) : 1;

													  //if (m_eta == 1)
													  //	FSAssert(u0 == u0_external  &&  uL == uL_external);

													  //if (optix::length(u0) == 0 || optix::length(uL) == 0) {
													  //	if (m_eta > 1)
													  //		Log(EWarn, "Could not refract, which is weird because we have a "
													  //			"higher ior! (eta=%f)", m_eta);
													  //	return 0.0f;
													  //}

	optix_print("Start.\n");
	Float3 R_virt;
	Float3 u0_virt;
	if (!getVirtualDipoleSource(material.sigma_s, material.sigma_a, material.mu, material.m_eta, n0, u0, nL, uL, R, length,
		props.rejectInternalIncoming, props.tangentMode, props.zvMode,
		u0_virt, R_virt, nullptr))
		return 0.0f;

	optix_print("Virtual source... R_virt %f %f %f, u_o %f %f %f\n", R_virt.x, R_virt.y, R_virt.z, u0_virt.x, u0_virt.y, u0_virt.z);
	// Effective BRDF?
	if (props.useEffectiveBRDF) {
		FSAssert(optix::length(n0 - nL) < Epsilon); // same point -> same Float3
		Float Rv_z = dot(R_virt, nL);
#ifdef MTS_FWDSCAT_DEBUG
		Float lRvl = optix::length(R_virt);
		FSAssert(optix::length(n0 - nL) < Epsilon); // same point -> same Float3
		FSAssert(Rv_z <= 0); // pointing from virtual point towards xL -> into medium
							 // the only displacement should be in the Float3 direction:
		FSAssertWarn(lRvl == 0 || abs((lRvl - abs(Rv_z)) / lRvl) < Epsilon);
#endif

		return (
			evalPlaneSource(material.sigma_s, material.sigma_a, material.mu, u0, uL, nL, 0.0f, length)
			- evalPlaneSource(material.sigma_s, material.sigma_a, material.mu, u0_virt, uL, nL, Rv_z, length));
	}

	// Full BSSRDF

	Float real = 0, virt = 0;
	if (props.dipoleMode & EReal)
		real = evalMonopole(material, u0, uL, R, length);
	if (props.dipoleMode & EVirt)
		virt = evalMonopole(material, u0_virt, uL, R_virt, length);
	Float transport;
	switch (props.dipoleMode) {
	case ERealAndVirt: transport = real - virt; break;
	case EReal:        transport = real; break;
	case EVirt:        transport = virt; break; // note: positive sign
	default: Log(EError, "Unknown dipoleMode: %d", props.dipoleMode); return 0;
	}

	optix_print("BSSRDF: %e\n", transport);

	if (props.reciprocal) {
		Float transportRev = evalDipole(material, props,
			nL, -uL, n0, -u0, -R, length);
		return 0.5 * (transport + transportRev);
	}
	else {
		return transport;
	}
}
#endif

__host__ __device__ __inline__ void test_forward_dipole_cuda()
{
	const Float3 xi = MakeFloat3(0., 0., 0.);
	const Float3 xo = MakeFloat3(0.03, 0., 0.);
	const Float3 ni = MakeFloat3(0., 0., 1.);
	const Float3 no = MakeFloat3(0., 0., 1.);
	const Float3 d_in = MakeFloat3(0., 0., -1.);
	const Float3 d_out = MakeFloat3(0., 0., 1.);

	const Float3 d_in_refr = MakeFloat3(0., 0., -1.);
	const Float3 d_out_refr = MakeFloat3(0., 0., 1.);
	
	ForwardDipoleMaterial material;
	material.sigma_s = 400.0f;
	material.sigma_a = 10.0f;
	material.mu = 1 - 0.9f;
	material.m_eta = 1.0f;
	Float3 R = xo - xi;
	unsigned int seed = 1023;
	Float s_test = 0.03;

	ForwardDipoleProperties props;
	props.dipoleMode = ERealAndVirt;
	props.reciprocal = false;
	props.rejectInternalIncoming = false;
	props.tangentMode = TangentPlaneMode::EFrisvadEtAl;
	props.useEffectiveBRDF = false;
	props.zvMode = ZvMode::EFrisvadEtAlZv;

	Float S = evalDipole(material, props,
		ni, d_in_refr,
		no, d_out_refr,
		R, s_test);

	Float ss = 0.0f;
	int N = 10;
	optix_print("%f %f %f\n", rnd_tea(seed), rnd_tea(seed), rnd_tea(seed));
	for (int i = 0; i < N; i++)
	{
		Float sss = 0.0;
		sampleLengthDipole(material, props, d_out, no, R, &d_in, ni, sss, seed);
		ss += sss;
	}

	optix_print("Dipole test: %e %e\n ", S, ss / N);
}

namespace optix
{
    __device__ __forceinline__ optix::float3 make_float3(const optix::float3 & c) { return c; }

}
__device__ __forceinline__ float3 forward_dipole_bssrdf(const BSSRDFGeometry & geometry, const float recip_ior, const MaterialDataCommon& material, unsigned int flags = BSSRDFFlags::NO_FLAGS, TEASampler * sampler = nullptr)
{
    const ScatteringMaterialProperties& properties = material.scattering_properties;
    float3 w12, w21;
    float R12, R21;
    bool include_fresnel_out = (flags &= BSSRDFFlags::EXCLUDE_OUTGOING_FRESNEL) == 0;
    refract(geometry.wi, geometry.ni, recip_ior, w12, R12);
    refract(geometry.wo, geometry.no, recip_ior, w21, R21);
    float F = include_fresnel_out? (1 - R21) : 1.0f;
    w21 = -w21;

	Float3 R = MakeFloat3(geometry.xo - geometry.xi);
	optix::float3 res;

#ifdef __CUDACC__
	unsigned int seed = tea<16>(launch_index.x, launch_index.y);
#else
	unsigned int seed = 1023;
#endif
	unsigned int samples = 1;

	const Float3 n_in = MakeFloat3(normalize(geometry.ni));
	const Float3 n_out = MakeFloat3(normalize(geometry.no));
	const Float3 d_in = MakeFloat3(w12);
	const Float3 d_out = MakeFloat3(w21);

	for (int k = 0; k < 3; k++)
	{

		ForwardDipoleMaterial material;
		material.sigma_s = optix::get_channel(k, properties.scattering);
		material.sigma_a = optix::get_channel(k, properties.absorption);
		material.mu = 1 - optix::get_channel(k, properties.meancosine);;
		material.m_eta = 1.0f;

		ForwardDipoleProperties props;
		props.dipoleMode = ERealAndVirt;
		props.reciprocal = false;
		props.rejectInternalIncoming = false;
		props.tangentMode = TangentPlaneMode::EFrisvadEtAl;
		props.useEffectiveBRDF = false;
		props.zvMode = ZvMode::EFrisvadEtAlZv;
		float S = 0.0f;
		for (unsigned int i = 0; i < samples; i++)
		{
			Float s = 0.0f;
			float wi = sampleLengthDipole(material, props, d_out, n_out, R, &d_in, n_in, s, seed);
			S += evalDipole(material, props,
				n_in, d_in,
				n_out, d_out,
				R, s) * wi;
		}
		optix::get_channel(k, res) = S / samples;
	}
	return res * (1 - R12) * F;
}
