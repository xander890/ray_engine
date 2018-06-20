#pragma once
#include "random_device.h"
#include "forward_dipole_utils.h"
#include "forward_dipole_truncnorm.h"
#include <functional>
#include "forward_dipole_tom748.h"
#include "sampler_device.h"


/**
* \brief Sample based purely on the absorption factor
*
* This is the safest bet 'at infinity' (the tail is certainly more heavy
* than the target distribution), but extremely high variance is possible
* for high albedo materials. */
_fn   Float sampleLengthAbsorption(const float sigma_a,
	Float &s, TEASampler * sampler) {
	if (sigma_a == 0)
		return 0.0;
	s = -log(sampler->next1D()) / sigma_a;
	Float pdf = sigma_a*exp(-sigma_a*s);
	FSAssert(isfinite(s));
	FSAssert(s >= 0);
	FSAssert(isfinite(pdf));
	return pdf;
}

_fn   Float pdfLengthAbsorption(const float sigma_a,
	Float s) {
	if (sigma_a == 0)
		return 0.0;
	Float pdf = sigma_a*exp(-sigma_a*s);
	FSAssert(isfinite(pdf));
	return pdf;
}

_fn   void implLengthShortLimitKnownU0(
	const Float sigma_s,
	const Float mu,
	Float3 R, Float3 u0, Float3 uL, Float &s, TEASampler * sampler, Float *pdf, bool use_sampler) {
	double p = 0.5*sigma_s*mu;
	double lRl = optix::length(R);
	double r = lRl * p;
	if (r == 0) {
		if (pdf) *pdf = 0;
		return;
	}
	double cosTheta0L = clamp(dot(R, u0) / lRl, -1.0, 1.0)
		+ clamp(dot(R, uL) / lRl, -1.0, 1.0);
	double u0dotuL = dot(u0, uL);

	double mean;
	if (r > 1e-4) { // full expression is sufficiently stable
					// transformation t = (ps)^(-3)
					// compute mean of gaussian in t: (root of a cubic polynomial)
					// Based on Maple codegen...
		double t1 = 0.1e1 / r;
		double t2 = cosTheta0L * cosTheta0L;
		double t3 = t2 * cosTheta0L;
		double t5 = sqrt(0.3e1);
		double t8 = u0dotuL * u0dotuL;
		double t18 = r * r;
		double t25 = -108 * r * u0dotuL * cosTheta0L + 96 * t3 * r
			- 216 * r * cosTheta0L - 4 * t2 * t8 - 16 * t2 * u0dotuL
			+ 4 * t8 * u0dotuL + 243 * t18 - 16 * t2 + 24 * t8
			+ 48 * u0dotuL + 32;
		double t26 = sqrt(t25);
		double t34 = cbrt(12 * t26 * t5 - (72 * cosTheta0L * u0dotuL)
			+ (324 * r) + (64 * t3) - (144 * cosTheta0L));
		double t35 = t34 * t1;
		double t42 = 1 / t34 * t1 * (-4 * t2 + 3 * u0dotuL + 6);
		double t44 = cosTheta0L * t1;
		double t46 = t35 / 18 + 2.0 / 9.0 * (t44 - t42);
		double t47 = t46 * t46;
		mean = 1.0 / 9.0 / t18 * (6 * cosTheta0L * t47 * r - u0dotuL * t46
			- t35 / 9.0 + 4.0 / 9.0*(t42 - t44) + 1);
	}
	else { // short r limit
		   // first nontrivial order expansion:
		double t1 = sqrt(3.0);
		double t3 = (u0dotuL + 2) * (u0dotuL + 2);
		double t4 = cosTheta0L * cosTheta0L;
		double t7 = sqrt(t3 * (-t4 + u0dotuL + 2));
		double t14 = 24 * t1 * t7 - 72 * cosTheta0L * (-8.0 / 9.0 * t4 + u0dotuL + 2);
		double t15 = cbrt(t14);
		double t16 = t15 * t15;
		double t28 = -8.0 / 3.0 * t4 + u0dotuL + 2.0;
		double t35 = t4 * t4;
		double t41 = u0dotuL * u0dotuL;
		double t48 = r * r;
		mean = ((48 * t4 * cosTheta0L + (-36 * u0dotuL - 72) * cosTheta0L) * t16
			+ 36 * (-4.0 / 3.0 * t4 + u0dotuL + 2) * t28 * t15
			- 72 * t1 * t28 * t7
			+ cosTheta0L * ((768 * t35) + ((-1152 * u0dotuL - 2304) * t4)
				+ t15 * t14 + (360 * t41) + (1440 * u0dotuL) + 1440))
			/ (t16 * t48 * r * 486);
	}
	if (!isfinite(mean) || mean <= 0) {
		/* This usually happens for small to negative u0dotuL and
		* cosTheta0L -- at which point there is no large ballistic peak
		* anyway!
		* Any choice is better than no choice, so set it as: */
		mean = 1. / (r*r*r); // 'pushing s to r'
	}
	FSAssert(isfinite(mean));
	FSAssert(mean > 0);

	double mean113 = pow(mean, 11. / 3.);
	double mean53 = pow(mean, 5. / 3.);
	double mean73 = pow(mean, 7. / 3.);
	double mean2 = mean*mean;
	double realStddev;
	if (r < 1e-4) {
		// short r limit expansion
		realStddev = sqrt((-54 * r * cosTheta0L + 12 * u0dotuL * u0dotuL + 48 * u0dotuL + 48) * pow(mean, 8. / 3.) / 27
			+ (18 * u0dotuL + 36) * mean73 / 27 + (8 * u0dotuL*u0dotuL*u0dotuL + 48 * u0dotuL * u0dotuL
				+ (-72 * r * cosTheta0L + 96) * u0dotuL - 144 * r * cosTheta0L + 64) * mean*mean*mean / 27 + mean * mean);
	}
	else {
		realStddev = sqrt((3 * mean113)
			/ (3 * mean53 + 6 * mean73 * r * cosTheta0L - (2 * u0dotuL + 4)*mean2));
	}
	double stddevSafetyFactor = 2;
	double stddev = stddevSafetyFactor * realStddev;
	if (!isfinite(stddev) || stddev <= 0) {
		stddev = mean; // heurstic!
	}
	FSAssert(isfinite(stddev));
	FSAssert(stddev>0);

	Float t, ps;

	if (use_sampler) {
		do {
			t = truncnorm(mean, stddev, 0.0, INFINITY, sampler);
		} while (t == 0);
		ps = powf(t, -1. / 3.);
		s = ps / p;
	}
	else {
		ps = p*s;
		t = 1 / (ps*ps*ps);
	}
	FSAssert(isfinite(s));
	FSAssert(s > 0);


	if (pdf) {
		Float tPdf = truncnormPdf(mean, stddev, 0.0, INFINITY, t);

		// transform from pdf(t = (ps)^(-3)) to pdf(ps) [factor 3*(ps)^-4] & go back to p!=1 [factor p]
		*pdf = tPdf * 3 / (ps*ps*ps*ps) * p;
	}
}

_fn   void implLengthShortLimitMargOverU0(
	const Float sigma_s,
	const Float mu,
	Float3 R, Float3 uL, Float &s, TEASampler * sampler, Float *pdf, bool use_sampler) {
	// Working in p=1, transforming back at the end
	Float p = 0.5*sigma_s*mu;
	Float lRl = optix::length(R);
	Float r = lRl * p;
	Float r2 = r*r;
	Float cosTheta = clamp(dot(R, uL) / lRl, (Float)-1, (Float)1);


	/* TODO:
	*
	* (1) This is not very sensible for r > 1 (set this strategy's MIS
	* weight to 0 then?)
	*
	* (2) The case r=0 can happen for an effective BRDF -> handle that
	* better by relaxing the cosTheta=1 assumption in derivation? Even
	* better: make dedicated sampler (which will have different s
	* behaviour, presumably) */
	//if (r == 0 || r > 1) {
	if (r == 0) {
		if (pdf) *pdf = 0;
		return;
	}

#if 1 /* Exact, fully cosTheta-dependent solution from Maple codegen (true), or 
					  crude, easy to evaluate approximation (false) */
					  // Maple codegen for (the real part of) the root that we want
	Float invps_mean; // the critical point (t* in the suppl. mat. of the paper)
	double t1 = 1. / r;
	double t5 = cosTheta * cosTheta;
	double t6 = cosTheta * t5;
	double t7 = t6 * r;
	double t9 = t5 * t5;
	double t11 = t5 * r;
	double t14 = r * r;
	double t16 = r * cosTheta;
	double t20 = 96 * t7 - 4 * t9 + 180 * t11 - 20 * t6 + 243 * t14
		- 36 * t16 - 28 * t5 - 120 * r + 16;


	double t21 = abs(t20);
	double t22 = sqrt(t21);
	double t23 = (t20 > 0) - (t20 < 0); // =signum(t20): (t20>0 ? 1 : (t20<0 ? -1 : 0));
	double t24 = t23 * t22;
	double t25 = sqrt(30.0);
	double t34 = t25 * t22;

	double t45 = -288 * cosTheta * t25 * t24 + 3888 * r * t34 - 960 * t23 * t34
		+ 1440 * t5 * t34 + 768 * t6 * t34 - 288 * cosTheta * t34 + 77760 * t11
		- 15552 * t16 + 216 * t21 + 41472 * t7 + 3840 * cosTheta + 6400;
	double t65 = t23 * t23;
	double t68 = 3888 * r * t25 * t24 + 1440 * t5 * t25 * t24
		+ 768 * t6 * t25 * t24 + 216 * t65 * t21 + 4096 * t5 * t9
		+ 15360 * cosTheta * t9 - 51840 * r + 104976 * t14
		- 960 * t34 - 18624 * t5 - 16000 * t6 + 11328 * t9;
	double t70 = pow(t45 + t68, 1.0 / 6.0);
	double t83 = atan2(6 * (1 - t23) * t34, (64 * t6) + 6 * (1 + t23) * t34
		+ (120 * t5) + (324 * r) - 24 * cosTheta - 80);
	double t85 = cos(t83 / 3);
	invps_mean = 2. / 9. * cosTheta * t1 + 2. / 9. * t1 + t85 * t70 * t1 / 18
		+ t85 / t70 * t1 * (16 * t5 + 20 * cosTheta - 8) / 18;
	// Note: t70 can give problems for negative values!
	if (!isfinite(invps_mean) || invps_mean <= 0) {
		invps_mean = 1 / r; // Heuristic 'guess' to not 'waste' a sample
	}
#else
					  /* Simplest possible choice. This is accurate up to 10% relative
					  * accuracy (and becomes more accurate for r->0). */
	Float invps_mean = 1 / r;
#endif

	FSAssert(isfinite(invps_mean));
	FSAssert(invps_mean > 0);
	Float t2 = invps_mean*invps_mean;
	Float t3 = invps_mean*t2;
	Float var = t2 / (3 + 54 * t3*r2 - 18 * r*(cosTheta + 1)*t2);

	if (!isfinite(var) || var <= 0) {
		/* This happens when we aren't in a local maximum, but a
		* *minimum* (var < 0)!
		* We can bail, or set the stddev to something 'safe' ... e.g. set
		* stddev=t just to sample *something* at least.
		* Probably better: determine suitability beforehand and don't use
		* this technique if it doesn't make sense. */
		var = invps_mean*invps_mean; // just some heuristic 'guess'
	}

	const Float stddevSafetyFactor = 2.5;
	Float stddev = stddevSafetyFactor * sqrt(var);

	Float invps;
	Float ps;
	if (use_sampler) {
		do {
			invps = truncnorm(invps_mean, stddev, 0.0, INFINITY, sampler);
		} while (invps == 0);
		ps = 1 / invps;
		s = ps / p;
	}
	else {
		ps = p*s;
		invps = 1 / ps;
	}

	FSAssert(isfinite(s));
	FSAssert(s > 0);

	if (pdf != nullptr) {
		Float invpsPdf = truncnormPdf(invps_mean, stddev, 0.0, FLT_MAX, invps);

		// transform from pdf(1/(ps)) to pdf(ps) [factor (ps)^-2] & go back to p!=1 [factor p]
		*pdf = invpsPdf / (ps*ps) * p;
	}
}

_fn   void implLengthShortLimit(
	const Float sigma_s,
	const Float mu,
	Float3 R, const Float3 *u0, Float3 uL, Float &s, TEASampler * sampler, Float *pdf, bool use_sampler) {
	if (u0 == nullptr) {
		implLengthShortLimitMargOverU0(sigma_s, mu, R, uL, s, sampler, pdf, use_sampler);
	}
	else {
		implLengthShortLimitKnownU0(sigma_s, mu, R, *u0, uL, s, sampler, pdf, use_sampler);
	}
}
_fn Float sampleLengthShortLimit(
	const Float sigma_s,
	const Float mu,
	Float3 R, const Float3 *u0, Float3 uL, Float &s, TEASampler * sampler) {
	Float pdf;
	implLengthShortLimit(sigma_s, mu, R, u0, uL, s, sampler, &pdf, true);
	return pdf;
}



_fn   Float pdfLengthShortLimit(
	const Float sigma_s,
	const Float mu,
	Float3 R, const Float3 *u0, Float3 uL, Float s) {
	Float pdf;
	implLengthShortLimit(sigma_s, mu, R, u0, uL, s, nullptr, &pdf, false);
	return pdf;
}

_fn   Float pdfLengthLongLimit(
	const Float sigma_s,
	const Float sigma_a,
	const Float mu,
	Float3 R, Float3 uL, Float s) {
	Float p = 0.5*sigma_s*mu;
	if (p == 0)
		return 0;
	Float s_p1 = s * p;
	Float3 R_p1 = R*p;
	Float R2minusRdotUL_p1 = dot(R_p1, R_p1) - dot(R_p1, uL);
	Float beta = 3. / 2. * R2minusRdotUL_p1;
	if (beta <= 0)
		return pdfLengthAbsorption(sigma_a, s);
	Float a_p1 = sigma_a / p;
	Float pdf_p1 = sqrt(beta / M_PI) / (s_p1*sqrt(s_p1))
		* exp(-beta / s_p1 - a_p1*s_p1 + 2 * sqrt(beta*a_p1));
	if (!isfinite(pdf_p1)) {
		//Log(EWarn, "FIXME %f %e %e %e", pdf_p1, beta, a_p1, s_p1);
		return 0;
	}
	return pdf_p1 * p;
}



// TODO: approximation that does not require a numerical cdf inversion?
_fn   Float sampleLengthLongLimit(
	const Float sigma_s,
	const Float sigma_a,
	const Float mu,
	Float3 R, Float3 uL, Float &s, TEASampler * sampler) {
	Float p = 0.5*sigma_s*mu;
	if (p == 0)
		return 0;
	Float3 R_p1 = R*p;
	optix_print("p %f, R %f %f %f, Ul %f %f %f\n", p, R.x, R.y, R.z, uL.x, uL.y, uL.z); 
	Float R2minusRdotUL_p1 = dot(R_p1, R_p1) - dot(R_p1, uL);
	Float beta = 3. / 2. * R2minusRdotUL_p1;
	if (beta <= 0)
		return sampleLengthAbsorption(sigma_a, s, sampler);
	double B = beta;
	double A = sigma_a / p;
	FSAssert(A>0);
	FSAssert(B>0);
	double sA = sqrt(A);
	double sB = sqrt(B);
	double C = exp(4 * sA*sB);
	auto cdf = [=](double ps) {
		double erfDiffArg = (sA*ps + sB) / sqrt(ps);
		double erfSumArg = (sA*ps - sB) / sqrt(ps);
		double erfDiff, erfSum;
		// expansion
		if (erfDiffArg > 3) {
			double x = erfDiffArg;
			double x2 = x*x;
			double x3 = x2*x;
			double x5 = x3*x2;
			erfDiff = (1 / x - 0.5 / x3 + .75 / x5)*exp(4 * sA*sB - x2) / sqrt(M_PI);
		}
		else {
			erfDiff = C * (1 - USE_ERF(erfDiffArg));
		}
		if (erfSumArg < -3) {
			double x = erfSumArg;
			double x2 = x*x;
			double x3 = x2*x;
			double x5 = x3*x2;
			erfSum = (-1 / x + 0.5 / x3 - .75 / x5) / exp(x2) / sqrt(M_PI);
		}
		else {
			erfSum = 1 + USE_ERF(erfSumArg);
		}
		double theCdf = 0.5*(erfDiff + erfSum);
		if (theCdf <= -Epsilon || theCdf >= 1 + Epsilon) {
			Log(EWarn, "invalid cdf: %e %e %e %e", theCdf, erfDiff, erfSum, C);
		};
		theCdf = clamp(theCdf, 0., 1.);
		return theCdf;
	};
	double u = sampler->next1D();
	auto target = [=](double ps) { return cdf(ps) - u; };
	optix_print("\ncdf(0.5) %f, u %f (sa %f sb %f c %f r2minus %f)\n", cdf(0.5f), u, sA, sB, C, R2minusRdotUL_p1);
	// Bracket the root
	double lo = 0;
	if (!isfinite(target(lo)) || target(lo) > 0) {
		Log(EWarn, "target(lo) did something weird: %f", target(lo));
		return 0;
	}
	double hi = 1000 / A;
	if (!isfinite(target(hi))) {
		Log(EWarn, "target(hi) not finite: %f", target(hi));
		return 0;
	}
	while (target(hi) < 0 && hi < 1e4 * 1000 / A)
		hi *= 3; // look further if we don't have the zero crossing bracketed
	if (!isfinite(target(hi)) || target(hi) < 0) {
		Log(EWarn, "could not find suitable target(hi): %f", target(hi));
		return 0;
	}

	size_t max_iter = 1000;
	optix::double2 Rvnsol = toms748_solve(target, lo, hi, eps_tolerance<double>(15), max_iter);
	Float s_p1 = 0.5*(Rvnsol.x + Rvnsol.y);
	s = s_p1 / p;
	if (!isfinite(s)) {
		Log(EWarn, "FIXME %f", s);
		return 0;
	}
	

	return pdfLengthLongLimit(sigma_s, sigma_a, mu,R, uL, s);
}




// Strategy weights, must sum to one
#define lengthSample_w1 0.0 /* short length limit */
#define lengthSample_w2 1.0 /* long length limit */
#define lengthSample_w3 0.0 /* absorption */

// If d_in is unknown, it is set to NULL
_fn  Float sampleLengthDipole(
	const ForwardDipoleMaterial material,
	const ForwardDipoleProperties props,
	Float3 uL, Float3 nL, Float3 R,
	const Float3 *u0, Float3 n0,
	Float &s, TEASampler * sampler) {

	Float3 R_virt;

	if (!getTentativeIndexMatchedVirtualSourceDisp(material.sigma_s, material.sigma_a, material.mu, material.m_eta,
		n0, nL, uL, R, NAN, props.tangentMode, R_virt))
		return 0.0;

	/* For R-dependent functions that don't take the dipole into account
	* themselves.
	* TODO: Smart MIS weight? (Need length-marginalized 'realSourceWeight'
	* from getTentativeIndexMatchedVirtualSourceDisp then.) */
	Float3 R_effective, R_other;
	if (sampler->next1D() < 0.5) {
		R_effective = R;
		R_other = R_virt;
	}
	else {
		R_effective = R_virt;
		R_other = R;
	}
	Float p1, p2, p3;
	p1 = p2 = p3 = -1;
	const Float u = sampler->next1D();
	if (u < lengthSample_w1) {
		p1 = sampleLengthShortLimit(material.sigma_s, material.mu, R, u0, uL, s, sampler);
		if (p1 == 0)
			return 0.0f;
	}
	else if (u < lengthSample_w1 + lengthSample_w2) {
		p2 = sampleLengthLongLimit(material.sigma_s, material.sigma_a, material.mu, R_effective, uL, s, sampler);
		if (p2 == 0)
			return 0.0f;
	}
	else if (u < lengthSample_w1 + lengthSample_w2 + lengthSample_w3) {
		p3 = sampleLengthAbsorption(material.sigma_a, s, sampler);
		if (p3 == 0)
			return 0.0f;
	}
	
	if (p1 == -1)
		p1 = (lengthSample_w1 == 0 ? 0 : pdfLengthShortLimit(material.sigma_s, material.mu, R, u0, uL, s));
	if (p2 == -1)
		p2 = (lengthSample_w2 == 0 ? 0 : pdfLengthLongLimit(material.sigma_s, material.sigma_a, material.mu, R_effective, uL, s));
	if (p3 == -1)
		p3 = (lengthSample_w3 == 0 ? 0 : pdfLengthAbsorption(material.sigma_a, s));

	// Handle the MIS probabilities of having sampled based on R_other
	if (lengthSample_w2 != 0)
		p2 = 0.5 * (p2 + pdfLengthLongLimit(material.sigma_s, material.sigma_a, material.mu, R_other, uL, s));

	return 1.0 / (lengthSample_w1 * p1
		+ lengthSample_w2 * p2
		+ lengthSample_w3 * p3);
}

_fn  Float pdfLengthDipole(
	const float sigma_s,
	const float sigma_a,
	const float mu,
	const float m_eta,
	const Float3 &uL, const Float3 &nL, const Float3 &R,
	const Float3 *u0, const Float3 &n0,
	TangentPlaneMode tangentMode, Float s) {
	FSAssert(s >= 0);
	Float3 R_virt;
	if (!getTentativeIndexMatchedVirtualSourceDisp(sigma_s, sigma_a, mu, m_eta,
		n0, nL, uL, R, NAN, tangentMode, R_virt))
		return 0.0;

	Float p1 = (lengthSample_w1 == 0 ? 0 :
		pdfLengthShortLimit(sigma_s, mu, R, u0, uL, s));
	Float p2 = (lengthSample_w2 == 0 ? 0 :
		0.5 * (pdfLengthLongLimit(sigma_s, sigma_a, mu,R, uL, s)
			+ pdfLengthLongLimit(sigma_s, sigma_a, mu,R_virt, uL, s)));
	Float p3 = (lengthSample_w3 == 0 ? 0 :
		pdfLengthAbsorption(sigma_a, s));
	return lengthSample_w1 * p1
		+ lengthSample_w2 * p2
		+ lengthSample_w3 * p3;
}