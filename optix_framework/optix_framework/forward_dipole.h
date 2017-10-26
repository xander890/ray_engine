#pragma once
#include "host_device_common.h"
#include "forward_dipole_sampling.h"

__device__ __host__ __forceinline__ Float evalMonopole(
	const Float sigma_s,
	const Float sigma_a,
	const Float mu,
	Float3 u0, Float3 uL, Float3 R, Float length) {
	FSAssert(abs(optix::length(u0) - 1) < 1e-6);
	FSAssert(abs(optix::length(uL) - 1) < 1e-6);

	double C, D, E, F;
	calcValues(length, sigma_s, sigma_a, mu, C, D, E, F);

	/* We regularized the sampling of u0, so we should be consistent here.
	* NOTE: but E can still blow up in the final expression for G (TODO
	* does this happen?) */
	// DOUBLE
	Float3 H = E * R - D*uL;
	double lHl = optix::length(H);
	Float3 Hnorm = Float3(H / lHl);
	Float lHlreg = (lHl > 1. / MTS_FWDSCAT_DIRECTION_MIN_MU) ?
		1. / MTS_FWDSCAT_DIRECTION_MIN_MU : lHl;
	Float cosTheta = roundCosThetaForStability(dot(u0, Hnorm), -1, 1);

	double N = absorptionAndFloat3izationConstant(length, sigma_s, sigma_a, mu);
	double G = N * exp(-C + E*dot(R, uL) + lHlreg*cosTheta - F*dot(R,R));
	//Non-regularized:
	//double G = N * exp(-C - D*dot(u0,uL) + E*(dot(R,u0) + dot(R,uL)) - F*R.lengthSquared());

	// Note: fastmath compiler flags may change the order of the operations...
	/* We only care for cancellations if the result is sufficiently large
	* (otherwise exp(epsilon) ~= 1 anyway) */
	if (abs(E*dot(R, uL)) > 1e3)
		CancellationCheck(-C, E*dot(R, uL));
	if (abs(lHlreg*cosTheta) > 1e3)
		CancellationCheck(-C + E*dot(R, uL), lHlreg*cosTheta);
	if (abs(F*dot(R,R)) > 1e3)
		CancellationCheck(-C + E*dot(R, uL) + lHlreg*cosTheta, -F*dot(R,R));

#ifdef MTS_FWDSCAT_DEBUG
	if (!isfinite(G) || G < 0) {
		Log(EWarn, "Invalid G in evalMonopole(): "
			"%e; s %e C %e D %e E %e F %e Rsq %e u0dotuL %e",
			G, length, C, D, E, F, dot(R,R), dot(u0, uL));
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

__device__ __host__ __forceinline__ Float evalDipole(
	const Float sigma_s,
	const Float sigma_a,
	const Float mu,
	const Float m_eta,
	Float3 n0, Float3 u0,
	Float3 nL, Float3 uL,
	Float3 R, Float length,
	bool rejectInternalIncoming,
	bool reciprocal,
	TangentPlaneMode tangentMode,
	ZvMode zvMode,
	bool useEffectiveBRDF = false ,
	DipoleMode dipoleMode = ERealAndVirt
	) {

	/* If reciprocal is requested, nL should be finite and uL_external should point
	* along nL. */
	FSAssert(!reciprocal || isfinite(nL));
	FSAssert(!reciprocal || dot(uL, nL) >= -Epsilon); // positive with small margin for roundoff errors
	//if (isfinite(nL) && dot(uL_external, nL) <= 0) // clamp to protect against roundoff errors
	//	return 0.0f;

#if MTS_FWDSCAT_DIPOLE_REJECT_INCOMING_WRT_TRUE_SURFACE_Float3
	//if (dot(u0_external, n0) >= 0)
	//	return 0.0f;
#endif


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
	optix_print("Getting virtual source...\n");
	if (!getVirtualDipoleSource(sigma_s, sigma_a, mu, m_eta, n0, u0, nL, uL, R, length,
		rejectInternalIncoming, tangentMode, zvMode,
		u0_virt, R_virt, nullptr))
		return 0.0f;

	// Effective BRDF?
	if (useEffectiveBRDF) {
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
			evalPlaneSource(sigma_s, sigma_a, mu,  u0, uL, nL, 0.0f, length)
			- evalPlaneSource(sigma_s, sigma_a, mu,  u0_virt, uL, nL, Rv_z, length));
	}

	// Full BSSRDF
	optix_print("Getting virtual source...\n");

	Float real = 0, virt = 0;
	if (dipoleMode & EReal)
		real = evalMonopole(sigma_s, sigma_a, mu, u0, uL, R, length);
	if (dipoleMode & EVirt)
		virt = evalMonopole(sigma_s, sigma_a, mu, u0_virt, uL, R_virt, length);
	Float transport;
	switch (dipoleMode) {
	case ERealAndVirt: transport = real - virt; break;
	case EReal:        transport = real; break;
	case EVirt:        transport = virt; break; // note: positive sign
	default: Log(EError, "Unknown dipoleMode: %d", dipoleMode); return 0;
	}

	optix_print("BSSRDF: %f\n", transport);

	if (reciprocal) {
		Float transportRev = evalDipole(sigma_s, sigma_a, mu, m_eta,
			nL, -uL, n0, -u0, -R, length,
			rejectInternalIncoming, false,
			tangentMode, zvMode, useEffectiveBRDF, dipoleMode);
		return 0.5 * (transport + transportRev);
	}
	else {
		return transport;
	}
}

__device__ __forceinline__ float3 forward_dipole_bssrdf(const float3& xi, const float3& ni, const float3& w12,
	const float3& xo, const float3& no, const float3& w21,
	const ScatteringMaterialProperties& properties)
{
	TangentPlaneMode tangent = TangentPlaneMode::EUnmodifiedIncoming;
	float3 R = xo - xi;
	optix::float3 res;
	unsigned int seed = 1023;
	unsigned int samples = 1000;

	for (int k = 0; k < 3; k++)
	{
		float sigma_s = optix::get_channel(k, properties.scattering);
		float sigma_a = optix::get_channel(k, properties.absorption);
		float mu = 1.0f - optix::get_channel(k, properties.meancosine);
		float eta = 1.0f;
		float S = 0.0f;

		for (int i = 0; i < samples; i++)
		{
			float s;
			float wi = sampleLengthDipole(sigma_s, sigma_a, mu, eta, w12, ni, R, nullptr, no, tangent, s, seed);
			S += evalDipole(sigma_s, sigma_a, mu, eta,
				no, normalize(w21),
				ni, normalize(w12),
				(xo - xi), s,
				false,
				false,
				tangent,
				ZvMode::EFrisvadEtAlZv,
				false,
				ERealAndVirt
			) * wi;
		}
		optix::get_channel(k, res) = S / samples;
	}
	return res;
}