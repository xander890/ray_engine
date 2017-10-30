#pragma once
#include "forward_dipole_utils.h"
#include "random.h"

#define RND_FUNC rnd_tea

__device__ __host__ __forceinline__ Float2 squareToStdNormal(const Float2 &sample) {
	Float r = sqrt(-2 * log(1 - sample.x)),
		phi = 2 * M_PI * sample.y;
	Float2 result = MakeFloat2(cos(phi), sin(phi));
	return result * r;
}

__device__ __host__ __forceinline__   Float stdnorm(unsigned int & t) {
	float xx = RND_FUNC(t);
	float yy = RND_FUNC(t);
	return squareToStdNormal(MakeFloat2(xx, yy)).x;
}


/// Check if simpler subalgorithm is appropriate.
__device__ __host__ __forceinline__   bool CheckSimple(const Float low, ///< lower bound of distribution
	const Float high ///< upper bound of distribution
) {
	// Init Values Used in Inequality of Interest
	Float val1 = (2 * sqrt(exp(1))) / (low + sqrt(pow(low, 2) + 4));
	Float val2 = exp((pow(low, 2) - low * sqrt(pow(low, 2) + 4)) / (4));
	//

	// Test if Simple is Preferred
	if (high > low + val1 * val2) {
		return true;
	}
	else {
		return false;
	}
}


/// XXX This check was missing from:
/// https://github.com/olmjo/RcppTN
/// http://olmjo.com/computing/RcppTN/
__device__ __host__ __forceinline__   bool CheckRejectFromUniformInsteadOfNormal(
	const Float low, const Float high) {
	if (low * high > 0)
		return false;
	return high - low < sqrt(2*M_PI);
}

/// Draw using algorithm 1.

/// 
/// Naive Accept-Reject algorithm.
/// 
/// Samples z from gaussian and rejects when out of bounds

__device__ __host__ __forceinline__   Float UseAlg1(const Float low, ///< lower bound of distribution
	const Float high, ///< upper bound of distribution
	unsigned int & t
) {
	// Init Valid Flag
	int valid = 0;
	//

	// Init Draw Storage
	Float z = 0.0;
	//

	// Loop Until Valid Draw
	while (valid == 0) {
		z = stdnorm(t);

		if (z <= high && z >= low) {
			valid = 1;
		}
	}
	//

	// Returns
	return z;
	//
}

/// Draw using algorithm 2.

/// 
///  Accept-Reject Algorithm
///
/// Samples from exponential distribution and rejects to transform to 
/// 'one-sided' bounded Gaussian.

__device__ __host__ __forceinline__   Float UseAlg2(const Float low, ///< lower bound of distribution
	unsigned int & t
) {
	// Init Values
	const Float alphastar = (low +
		sqrt(pow(low, 2) + 4.0)
		) / (2.0);
	const Float alpha = alphastar;
	Float z;
	Float rho;
	Float u;
	//

	// Init Valid Flag
	int valid = 0;
	//

	// Loop Until Valid Draw
	while (valid == 0) {
		Float e = -log(RND_FUNC(t));
		z = low + e / alpha;

		rho = exp(-pow(alpha - z, 2) / 2);
		u = RND_FUNC(t);
		if (u <= rho) {
			// Keep Successes
			valid = 1;
		}
	}
	//

	// Returns
	return z;
	//
}

/// Draw using algorithm 3.

/// 
/// Accept-Reject Algorithm
/// 
/// Samples z uniformly within lo..hi and rejects based on gaussian weight

__device__ __host__ __forceinline__   Float UseAlg3(const Float low, ///< lower bound of distribution
	const Float high, ///< upper bound of distribution
	unsigned int & t
) {
	// Init Valid Flag
	int valid = 0;
	//

	// Declare Qtys
	Float rho;
	Float z;
	Float u;
	//

	// Loop Until Valid Draw
	while (valid == 0) {
		z = low + RND_FUNC(t) * (high - low);
		if (0 < low) {
			rho = exp((pow(low, 2) - pow(z, 2)) / 2);
		}
		else if (high < 0) {
			rho = exp((pow(high, 2) - pow(z, 2)) / 2);
		}
		else {
			SAssert(0 <= high && low <= 0);
			rho = exp(-pow(z, 2) / 2);
		}

		u = RND_FUNC(t);
		if (u <= rho) {
			valid = 1;
		}
	}
	//

	// Returns
	return z;
	//
}

__device__ __host__ __forceinline__   Float truncnorm(const Float mean,
	const Float sd,
	const Float low,
	const Float high,
	unsigned int & t
) {
	if (low == high)
		return low;

	if (sd == 0) {
		if (low <= mean && mean <= high)
			return mean;
		if (mean > high)
			return high;
		return low;
	}

	if (isinf(sd)) {
		return low + RND_FUNC(t) * (high - low);
	}

	SAssert(sd > 0);

	// Init Useful Values
	Float draw = 0;
	int type = 0;
	int valid = 0; // used only when switching to a simplified version
				   // of Alg 2 within Type 4 instead of the less
				   // efficient Alg 3

				   // Set Current Distributional Parameters
	const Float c_mean = mean;
	Float c_sd = sd;
	const Float c_low = low;
	const Float c_high = high;
	Float c_stdlow = (c_low - c_mean) / c_sd;
	Float c_stdhigh = (c_high - c_mean) / c_sd; // bounds are standardized

	// Map Conceptual Cases to Algorithm Cases
	// Case 1 (Simple Deterministic AR)
	// mu \in [low, high]
	if (0 <= c_stdhigh &&
		0 >= c_stdlow
		) {
		type = 1;
	}

	// Case 2 (Robert 2009 AR)
	// mu < low, high = Inf
	if (0 < c_stdlow &&
		c_stdhigh == INFINITY
		) {
		type = 2;
	}

	// Case 3 (Robert 2009 AR)
	// high < mu, low = -Inf
	if (0 > c_stdhigh &&
		c_stdlow == -INFINITY
		) {
		type = 3;
	}

	// Case 4 (Robert 2009 AR)
	// mu -\in [low, high] & (abs(low) =\= Inf =\= high)
	if ((0 > c_stdhigh || 0 < c_stdlow) &&
		!(c_stdhigh == INFINITY || c_stdlow == -INFINITY)
		) {
		type = 4;
	}


	////////////
	// Type 1 //
	////////////
	if (type == 1) {
		if (CheckRejectFromUniformInsteadOfNormal(c_stdlow, c_stdhigh))
			draw = UseAlg3(c_stdlow, c_stdhigh, t);
		else
			draw = UseAlg1(c_stdlow, c_stdhigh, t);
	}

	////////////
	// Type 3 //
	////////////
	if (type == 3) {
		c_stdlow = -1 * c_stdhigh;
		c_stdhigh = INFINITY;
		c_sd = -1 * c_sd; // hack to get two negative signs to cancel out

						  // Use Algorithm #2 Post-Adjustments
		type = 2;
	}

	////////////
	// Type 2 //
	////////////
	if (type == 2) {
		draw = UseAlg2(c_stdlow, t);
	}

	////////////
	// Type 4 //
	////////////

	if (type == 4) {
		// Flip to make the standardized bounds positive if they aren't 
		// already, or else Alg2 fails
		SAssert(c_stdlow * c_stdhigh > 0); // double check that both have same sign
		if (c_stdlow < 0) {
			double tmp = c_stdlow;
			c_stdlow = -c_stdhigh;
			c_stdhigh = -tmp;
			c_sd = -1 * c_sd; // hack to get two negative signs to cancel out
		}

		if (CheckSimple(c_stdlow, c_stdhigh)) {
			while (valid == 0) {
				draw = UseAlg2(c_stdlow, t);
				// use the simple
				// algorithm if it is more
				// efficient
				if (draw <= c_stdhigh) {
					valid = 1;
				}
			}
		}
		else {
			draw = UseAlg3(c_stdlow, c_stdhigh, t); // use the complex
														  // algorithm if the simple
														  // is less efficient
		}
	}

	if (draw < c_stdlow || draw > c_stdhigh) {
		SLog(EWarn, "Generated out of bounds draw: %f not in [%f .. %f]",
			draw, c_stdlow, c_stdhigh);
	}
	return clamp(c_mean + c_sd * draw, low, high); // to protect against round-off
}

__device__ __host__ __forceinline__   Float truncnormPdf(const Float _mean,
	const Float _sd,
	const Float _lo,
	const Float _hi,
	const Float _z) {
	/* We do everything explicitly in doubles here, because the
	* exponentials are too prone to over/underflow with single precision
	* */
	double mean(_mean);
	double sd(_sd);
	double lo(_lo);
	double hi(_hi);
	double z(_z);

	SAssert(lo <= hi);

	if (z < lo || z > hi)
		return 0.0f;

	if (lo == hi)
		return 1.0f;

	if (sd == 0) {
		double scale = hi - lo;
		if (!isfinite(scale))
			SLog(EError, "I currently only support finite intervals when sd==0");
		double acceptedError = Epsilon * scale;
		if (lo <= mean && mean <= hi)
			return abs(z - mean) < acceptedError ? 1.0 : 0.0;
		if (mean > hi)
			return abs(z - hi) < acceptedError ? 1.0 : 0.0;
		return abs(z - lo) < acceptedError ? 1.0 : 0.0;
	}

	if (isinf(sd)) {
		return 1.0 / (hi - lo);
	}

	SAssert(sd > 0);

	/* If both erfs in the denominator go to one (meaning both
	* bounds are [far] to the right of the mean), we simply use the
	* mirror property to get both bounds to the left of the mean
	* where each erf is something small and there is less catastrophic
	* cancellation */
	if (lo >= mean && hi > mean) {
		SAssert(z >= mean);
		double loFlipped = mean - (hi - mean);
		double hiFlipped = mean - (lo - mean);
		double zFlipped = mean - (z - mean);
		lo = loFlipped;
		hi = hiFlipped;
		z = zFlipped;
		SAssert(lo < mean && hi <= mean && z <= mean);
	}
	SAssert(lo < mean);


	double pdf;
	double c_stdhi = (hi - mean) / sd; // standarized bound
	double c_stdlo = (lo - mean) / sd; // standarized bound 
	double c_stdz = (z - mean) / sd; // standarized sample
	if (c_stdhi > -8.) { // in this case: full erf expression should be sufficiently stable
		double absoluteExpArgument = 0.5 * ((z - mean) / sd) *  ((z - mean) / sd);
		double erf1 = (hi - mean) / (M_SQRT2f*sd);
		double erf2 = (lo - mean) / (M_SQRT2f*sd);
		double erfDiff = USE_ERF(erf1) - USE_ERF(erf2);
		//SAssert(absoluteExpArgument < LOG_REDUCED_PRECISION); // this can underflow if pdf becomes 0, which is OK...
		SAssert(erfDiff > 0);
		pdf = 2.0*exp(-absoluteExpArgument)
			/ ((sqrt(2 * M_PI) * sd) * erfDiff);
		if (!isfinite(pdf))
			Log(EWarn, "full pdf %e, %e %e %e %e", pdf, c_stdlo, c_stdhi, c_stdz, sd);
	}
	else {
		/* Our bounds are *waaaay* to the left of the mean, so the exponential
		* and erfs can potentially underflow. Expand the erfs and cancel
		* exp(lo^2/2) factors [note that exp(lo^2/2) > exp(hi^2/2), because
		* lo and hi are both <0, and lo<hi, so lo is bigger in absolute
		* value] */
		SAssert(c_stdlo < 0);
		SAssert(c_stdhi < 0);
		SAssert(c_stdz < 0);
		pdf = exp(0.5*(c_stdhi*c_stdhi - c_stdz*c_stdz)) * c_stdlo * c_stdhi
			/ (-c_stdlo + c_stdhi*exp(0.5*(c_stdhi*c_stdhi - c_stdlo*c_stdlo)));
		pdf /= sd; // transform back to non-standard setting
		if (!isfinite(pdf))
			Log(EWarn, "expanded pdf %e, %e %e %e %e", pdf, c_stdlo, c_stdhi, c_stdz, sd);
	}
	//SAssert(std::isfinite(pdf));
	return pdf;
}