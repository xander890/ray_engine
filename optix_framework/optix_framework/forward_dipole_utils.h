#pragma once
#include "forward_dipole_defines.h"
__host__ __device__ __forceinline__ double erf_approx_toshiya(double x) {
	const double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
	double t = 1.0 / (1.0 + p*abs(x));
	double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
	return copysign(1.0, x)*y;
}

#define USE_ERF erf

#ifdef MTS_WITH_CANCELLATION_CHECKS
__host__ __device__ __forceinline__ bool catastrophicCancellation(float a, float b) {
	return abs(a + b) / abs(a - b) < 1e-4;
}
__host__ __device__ __forceinline__ bool catastrophicCancellation(double a, double b) {
	return abs(a + b) / abs(a - b) < 1e-7;
}

# define CancellationCheck(a, b) do {\
	if (catastrophicCancellation(a,b))\
		Log(EWarn, "Catastrophic cancellation! (relative %e, %e & %e)",\
				abs((a)+(b))/abs((a)-(b)), a, b);\
	} while(0)
#else
# define CancellationCheck(a, b) ((void) 0)
#endif

__device__ __host__ __forceinline__ bool isfinited(Float3 & v)
{
    Float3 vv = v;
	return isfinite(vv.x) && isfinite(vv.y) && isfinite(vv.z);
}

template<typename T>
__host__ __device__ __forceinline__  T get_min();
template<>
__host__ __device__ __forceinline__ float get_min() { return FLT_MIN; }
template<>
__host__ __device__ __forceinline__ double get_min() { return DBL_MIN; }

template<typename T>
__host__ __device__ __forceinline__  T get_max();
template<>
__host__ __device__ __forceinline__ float get_max() { return FLT_MAX; }
template<>
__host__ __device__ __forceinline__ double get_max() { return DBL_MAX; }

template<typename T>
__host__ __device__ __forceinline__  T get_epsilon();
template<>
__host__ __device__ __forceinline__ float get_epsilon() { return FLT_EPSILON; }
template<>
__host__ __device__ __forceinline__ double get_epsilon() { return DBL_EPSILON; }

__device__ __host__ __forceinline__  Float dEon_C1(const Float n) {
	Float r;
	if (n > 1.0) {
		r = -9.23372 + n * (22.2272 + n * (-20.9292 + n * (10.2291 + n * (-2.54396 + 0.254913 * n))));
	}
	else {
		r = 0.919317 + n * (-3.4793 + n * (6.75335 + n *  (-7.80989 + n *(4.98554 - 1.36881 * n))));
	}
	return r / 2.0;
}

__device__ __host__ __forceinline__  Float dEon_C2(const Float n) {
	Float r = -1641.1 + n * (1213.67 + n * (-568.556 + n * (164.798 + n * (-27.0181 + 1.91826 * n))));
	r += (((135.926 / n) - 656.175) / n + 1376.53) / n;
	return r / 3.0;
}

__device__ __host__ __forceinline__  Float dEon_A(const Float eta) {
	return (1 + 3 * dEon_C2(eta)) / (1 - 2 * dEon_C1(eta));
}

__device__ __host__ __forceinline__ void calcValues(const double length, const Float sigma_s, const Float sigma_a, const Float mu, double &C, double &D, double &E, double &F) {
	FSAssert(length >= 0);
	FSAssert(mu > 0 && mu <= 1);
	FSAssert(sigma_s > 0);
	FSAssert(length >= 0);

	double p = 0.5 * mu * sigma_s;
	double s = length;
	double ps = p*s;

	/* TODO/NOTE: C is independent of R,u0,uL eventually, so purely
	* a Float3ization problem!
	* We could drop C, but that has the effect of exploding the
	* Float3ization constant exponential beyond double precision range for
	* small p*s. So we currently keep it as a help for numerical stability.
	* (Maybe a reduced form leads to something that is still managable and
	* allows for a simpler Float3ization function fit?) Either way, the
	* nu0 term in C is simply dropped */

	if (ps < 0.001) {
		/* Expansion accurate up to a range of 6 orders of ps */
		C = 3. / ps + 0.4*ps - 11. / 525.*ps*ps*ps;
		D = 1.5 / ps - 0.1*ps + 13. / 1050.*ps*ps*ps;
		E = p * (4.5 / (ps*ps) + 0.3 - 3. / 350 * ps*ps);
		F = p*p * (4.5 / (ps*ps*ps) + 1.8 / ps - 3. / 350 * ps);
	}
	else if (ps > 1.0 / 0.001) {
		/* Expansion accurate up to a range of 'all' orders of 1/ps (exact
		* geometric series) */
		double series = 1.0 / (ps - 1.0); // = 1/ps + 1/ps^2 + 1/ps^3 + ...
		C = 1.5 + 0.75 * series;
		D = 0.75 * series;
		E = p * 1.5 * series;
		F = p*p * 1.5 * series;
	}
	else {
		/* Exact solutions, in a ps range that is safe from numerical problems */
		double TH = tanh(p*s);
		double SH = sinh(2 * p*s);
		double TH2 = tanh(2 * p*s); /* Note: SH/CH with CH = sqrt(1 + SH*SH) is
									unstable for large arguments (inf/inf) */

		double A = 1 / (s / p - TH / (p*p));
		double B = TH / (2 * p);

		CancellationCheck(3 * A*B*B, 3 / (2 * TH2)); // C
		CancellationCheck((3 * A*B*B), (-3 / (2 * SH)));  // D
		C = 3 * A*B*B + 3 / (2 * TH2);
		D = 3 * A*B*B - 3 / (2 * SH);
		E = 3 * A*B;
		F = 3 * A / 2;
	}

	FSAssert(C >= 0);
	FSAssert(D >= 0);
	FSAssert(E >= 0);
	FSAssert(F >= 0);
}

__device__ __host__ __forceinline__ Float fresnelDiffuseReflectance(Float eta) {
	/* Fast mode: the following code approximates the
	* diffuse Frensel reflectance for the eta<1 and
	* eta>1 cases. An evalution of the accuracy led
	* to the following scheme, which cherry-picks
	* fits from two papers where they are best.
	*/
	if (eta < 1) {
		/* Fit by Egan and Hilgeman (1973). Works
		reasonably well for "normal" IOR values (<2).

		Max rel. error in 1.0 - 1.5 : 0.1%
		Max rel. error in 1.5 - 2   : 0.6%
		Max rel. error in 2.0 - 5   : 9.5%
		*/
		return -1.4399f * (eta * eta)
			+ 0.7099f * eta
			+ 0.6681f
			+ 0.0636f / eta;
	}
	else {
		/* Fit by d'Eon and Irving (2011)
		*
		* Maintains a good accuracy even for
		* unrealistic IOR values.
		*
		* Max rel. error in 1.0 - 2.0   : 0.1%
		* Max rel. error in 2.0 - 10.0  : 0.2%
		*/
		Float invEta = 1.0f / eta,
			invEta2 = invEta*invEta,
			invEta3 = invEta2*invEta,
			invEta4 = invEta3*invEta,
			invEta5 = invEta4*invEta;

		return 0.919317f - 3.4793f * invEta
			+ 6.75335f * invEta2
			- 7.80989f * invEta3
			+ 4.98554f * invEta4
			- 1.36881f * invEta5;
	}
}

__device__ __host__ __forceinline__ Float fresnelDielectricExt(Float cosThetaI_, Float &cosThetaT_, Float eta) {
	if (eta == 1) {
		cosThetaT_ = -cosThetaI_;
		return 0.0f;
	}

	/* Using Snell's law, calculate the squared sine of the
	angle between the normal and the transmitted ray */
	Float scale = (cosThetaI_ > 0) ? 1 / eta : eta,
		cosThetaTSqr = 1 - (1 - cosThetaI_*cosThetaI_) * (scale*scale);

	/* Check for total internal reflection */
	if (cosThetaTSqr <= 0.0f) {
		cosThetaT_ = 0.0f;
		return 1.0f;
	}

	/* Find the absolute cosines of the incident/transmitted rays */
	Float cosThetaI = abs(cosThetaI_);
	Float cosThetaT = sqrt(cosThetaTSqr);

	Float Rs = (cosThetaI - eta * cosThetaT)
		/ (cosThetaI + eta * cosThetaT);
	Float Rp = (eta * cosThetaI - cosThetaT)
		/ (eta * cosThetaI + cosThetaT);

	cosThetaT_ = (cosThetaI_ > 0) ? -cosThetaT : cosThetaT;

	/* No polarization -- return the unpolarized reflectance */
	return 0.5f * (Rs * Rs + Rp * Rp);
}

__device__ __host__ __forceinline__ Float _reducePrecisionForCosTheta(Float x) {
	/* Turns out not to help too much -- or even make things worse! So
	* don't round. TODO: Test some more at some point... */
	return x;
	//return roundFloatForStability(x);
	//return roundToSignificantDigits(x, 3);
}

__device__ __host__ __forceinline__ double absorptionAndFloat3izationConstant(Float theLength, const Float sigma_s, const Float sigma_a, const Float mu) {
	double p = 0.5 * sigma_s * mu;
	double ps = p * theLength;

	double result;
	if (ps < 0.03) {
		double ps2 = ps*ps;
		double ps3 = ps2*ps;
		result = sqrt(2.0) * pow(M_PI, -2.5)
			* pow(ps, -11. / 2.) * (
				81. / 32. + 243. / 64.*ps + 3429. / 1280.*ps2 - 243. / 2560.*ps3);
		result *= p*p*p; // from p=1 back to the real p value;
		result *= exp(-sigma_a*theLength);
	}
	else if (ps > 11) { // quickly to avoid cancellation in denominator of full expression!
		double psi1 = 1.0 / ps;
		double psi2 = psi1*psi1;
		double psi3 = psi2*psi1;
		double psi4 = psi3*psi1;
		result = exp(1.5)*sqrt(6.0)*3. / 2048. * pow(M_PI, -2.5)
			* pow(ps, -3. / 2.) * (
				128 + 129 * psi1 + 240 * psi2 + 280 * psi3 + 315 * psi4);
		result *= p*p*p; // from p=1 back to the real p value;
		result *= exp(-sigma_a*theLength);
	}
	else {
		double C, D, E, F;
		calcValues(theLength, sigma_s, sigma_a, mu, C, D, E, F);
		double denomExpArg = E*E / F - 2 * D; /* cancellation problem, so quickly go to large ps
											  expansion TODO: rewrite in terms of A&B to avoid
											  cancellation */
		double denom = abs(denomExpArg) > 1e-3
			? exp(denomExpArg) - 1
			: denomExpArg*(1 + denomExpArg*(0.5 + denomExpArg*1. / 6.));
		result = 0.25 / pow(M_PI, 2.5)
			* sqrt(F) * (E*E - 2 * D*F) * exp(C - D - sigma_a*theLength)
			/ denom;
#ifdef MTS_FWDSCAT_DEBUG
		FSAssertWarn(isfinite(exp(D + C)));
		FSAssertWarn(isfinite(exp(2 * D)));
		FSAssertWarn(isfinite(exp(E*E / F)));
		// denom
		CancellationCheck(E*E / F, -2 * D); /* need to switch to large ps expansion
											sufficiently fast, see above */
											// exp in result:
		CancellationCheck(C, -D);
#endif
	}

#ifdef MTS_FWDSCAT_DEBUG
	if (!isfinite(result)) {
		Log(EWarn, "problem with analytical Float3ization at ps %e: %e",
			ps, result);
	}
#endif
	FSAssert(result >= 0);

	return result;
}



/// if rejectInternalIncoming is requested: returns false if we should stop
__device__ __host__ __forceinline__ bool getVirtualDipoleSource(
	const Float sigma_s, const Float sigma_a, const Float mu, const Float m_eta,
	Float3 n0, Float3 u0,
	Float3 nL, Float3 uL,
	Float3 R, Float length,
	bool rejectInternalIncoming,
	TangentPlaneMode tangentMode,
	ZvMode zvMode,
	Float3 &u0_virt, Float3 &R_virt,
	Float3 *optional_n0_effective = nullptr) {
	Float3 n0_effective;
    optix_print("L0 %f\n ", optix::length(n0));
	switch (tangentMode) {
	case EFrisvadEtAl:
		/* Use the modified tangent plane of the directional dipole model
		* of Frisvad et al */
		if (optix::length(R) == 0) {
			n0_effective = n0;
            optix_print("L0\n");
		}
		else {
			if (optix::length(cross(n0, R)) == 0)
				return false;
			n0_effective = normalize(cross(normalize(R), normalize(cross(n0, R))));
			FSAssert(dot(n0_effective, n0) > -Epsilon);
		}
		break;
	case EFrisvadEtAlWithMeanNormal: {
		/* Like the tangent plane of Frisvad et al, but based on an
		* 'average' Float3 at incoming and outgoing point instead of on
		* the incoming Float3. This should immediately give reciprocity as
		* a bonus. */
		Float3 sumFloat3 = n0 + nL;
		if (optix::length(R) == 0) {
			n0_effective = n0;
		}
		else {
			if (optix::length(cross(sumFloat3, R)) == 0)
				return false;
			n0_effective = normalize(cross(normalize(R), normalize(cross(sumFloat3, R))));
		}
		break; }
	case EUnmodifiedIncoming:
		n0_effective = n0; break;
	case EUnmodifiedOutgoing:
		n0_effective = nL; break;
	default:
		Log(EError, "Unknown tangentMode: %d", tangentMode);
	}
	if (!isfinited(n0_effective)) {
		Log(EWarn, "Non-finite n0_effective: %f %f %f", n0_effective.x, n0_effective.y, n0_effective.z);
		return false;
	}

	if (rejectInternalIncoming && dot(n0_effective, u0) > 0)
		return false;

	optix_print("%f\n", optix::length(n0_effective) - 1);
	FSAssert(abs(optix::length(n0_effective) - 1) < Epsilon);

	Float zv;
	Float sigma_sp = sigma_s * mu;
	Float sigma_tp = sigma_sp + sigma_a;
	
	switch (zvMode) {
	case EFrisvadEtAlZv: {
		if (sigma_tp == 0 || sigma_sp == 0)
			return false;
		Float D = 1. / (3.*sigma_tp);
		Float alpha_p = sigma_sp / sigma_tp;
		Float d_e = 2.131 * D / sqrt(alpha_p);
		Float A = dEon_A(m_eta);
		zv = 2 * A*d_e;
		break; }
	case EBetterDipoleZv: {
		if (sigma_tp == 0)
			return false;
		Float D = (2 * sigma_a + sigma_sp) / (3 * sigma_tp * sigma_tp);
		Float A = dEon_A(m_eta);
		zv = 4 * A*D;
		break; }
	case EClassicDiffusion: {
		if (sigma_tp == 0)
			return false;
		Float Fdr = fresnelDiffuseReflectance(1 / m_eta);
		Float A = (1 + Fdr) / (1 - Fdr);
		Float D = 1. / (3 * sigma_tp);
		zv = 4 * A*D;
		break; }
	default:
		Log(EError, "Unknown VirtSourceHeight mode %d", zvMode);
		return false;
	}

	/* If not rejectInternalIncoming -> virtual source will point *INTO*
	* the half space!! (and 'cross' the actual real source "beam" if we
	* elongate it).
	* Maybe flip the Float3? (to get the half space on the other side...) */
	//optix_print("Effective normal %f %f %f zv %f \n", n0_effective.x, n0_effective.y, n0_effective.z, zv);

	R_virt = R - zv * n0_effective;
	u0_virt = u0 - 2 * dot(n0_effective, u0) * n0_effective;
	if (optional_n0_effective)
		*optional_n0_effective = n0_effective;
	return true;
}

__device__ __host__ __forceinline__ bool getTentativeIndexMatchedVirtualSourceDisp(
	const Float sigma_s,
	const Float sigma_a,
	const Float mu,
	const Float m_eta,
	Float3 n0,
	Float3 nL, Float3 uL,
	Float3 R,
	Float s, // not always required
	TangentPlaneMode tangentMode,
	Float3 &R_virt,
	Float3 *optional_n0_effective = nullptr,
	Float *optional_realSourceRelativeWeight = nullptr) 
{
	Float3 _u0_virt, n0_effective;
	Float3 _u0 = MakeFloat3(NAN);
	bool rejectInternalIncoming = false; //u0 not sensible yet!
	ZvMode zvMode = EClassicDiffusion; //only one that does not depend on u0
	if (!getVirtualDipoleSource(sigma_s, sigma_a, mu, m_eta, n0, _u0, nL, uL, R, s,
		rejectInternalIncoming, tangentMode, zvMode,
		_u0_virt, R_virt, &n0_effective)) {
		return false; // Won't be able to evaluate bssrdf transport anyway!
	}
	else {
		FSAssert(isfinited(R_virt));
	}


	if (optional_n0_effective != nullptr)
		*optional_n0_effective = n0_effective;
	if (optional_realSourceRelativeWeight == nullptr)
		return true;
	double C, D, E, F;
	calcValues(s, sigma_s, sigma_a, mu, C, D, E, F);
	double ratio = exp(E*dot(R - R_virt, uL) - F*(dot(R, R) - dot(R_virt, R_virt)));
	Float realSourceWeight = (isinf(ratio + 1) ? 1.0 : ratio / (ratio + 1));
	// TODO: clamp the extremes of 0 and 1 to something slightly more 'centered'?
	FSAssert(realSourceWeight >= 0 && realSourceWeight <= 1);
#if MTS_FWDSCAT_GIVE_REAL_AND_VIRTUAL_SOURCE_EQUAL_SAMPLING_WEIGHT
	*optional_realSourceRelativeWeight = 0.5;
#else
	*optional_realSourceRelativeWeight = realSourceWeight;
#endif
	return true;
}

