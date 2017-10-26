// Copyright 2005 Mitsubishi Electric Research Laboratories All Rights Reserved.

// Permission to use, copy and modify this software and its documentation without
// fee for educational, research and non-profit purposes, is hereby granted, provided
// that the above copyright notice and the following three paragraphs appear in all copies.

// To request permission to incorporate this software into commercial products contact:
// Vice President of Marketing and Business Development;
// Mitsubishi Electric Research Laboratories (MERL), 201 Broadway, Cambridge, MA 02139 or 
// <license@merl.com>.

// IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL,
// OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
// ITS DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

// MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED
// HEREUNDER IS ON AN "AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
// UPDATES, ENHANCEMENTS OR MODIFICATIONS.
#ifndef BRDF_F
#define BRDF_F
#include <optix_world.h>
#include <optix_math.h>
#include <cmath>

#define BRDF_SAMPLING_RES_THETA_H       90
#define BRDF_SAMPLING_RES_THETA_D       90
#define BRDF_SAMPLING_RES_PHI_D         360

#define RED_SCALE (1.0f/1500.0f)
#define GREEN_SCALE (1.15f/1500.0f)
#define BLUE_SCALE (1.66f/1500.0f)

#include "sampling_helpers.h"
#include "random.h"
#include "math_helpers.h"



// rotate vector around an axis
static __host__  __device__ __inline__ optix::float3 rotate_vector(const optix::float3& vector, const optix::float3& axis, float angle)
{
  float sin_ang, cos_ang;
  sincosf(angle, &sin_ang, &cos_ang);

  optix::float3 out = vector*cos_ang;

  float temp = dot(axis, vector);
  temp *= (1.0f - cos_ang);
  out += axis*temp;
	
  optix::float3 cross_v = optix::cross(axis, vector);
  out += cross_v*sin_ang;
  return out;
}

// convert vectors in tangent space to half vector/difference vector coordinates
static __host__ __device__ __inline__ void vectors_to_half_diff_coords(
  const optix::float3& in, const optix::float3& out,
  float& theta_half, float& phi_half, float& theta_diff, float& phi_diff)
{
  // compute halfway vector
  optix::float3 half = normalize(in + out);

  // compute theta_half, phi_half
  theta_half = acosf(half.z);
  phi_half = atan2f(half.y, half.x);

  // compute diff vector
  const optix::float3 bi_normal = optix::make_float3(0.0, 1.0, 0.0);
  const optix::float3 normal = optix::make_float3(0.0, 0.0, 1.0);
  optix::float3 temp = rotate_vector(in, normal, -phi_half);
  optix::float3 diff = rotate_vector(temp, bi_normal, -theta_half);

  // compute theta_diff, phi_diff	
  theta_diff = acosf(diff.z);
  phi_diff = atan2f(diff.y, diff.x);
}


// Lookup theta_half index
// This is a non-linear mapping!
// In:  [0 .. pi/2]
// Out: [0 .. 89]
static __host__ __device__ __inline__ int theta_half_index(float theta_half)
{
  theta_half = fmaxf(theta_half, 0.0f);
  float theta_half_deg = theta_half*M_1_PIf*2.0f*BRDF_SAMPLING_RES_THETA_H;
  float temp = sqrt(theta_half_deg*BRDF_SAMPLING_RES_THETA_H);
  int idx = static_cast<int>(temp);
  return optix::min(idx, BRDF_SAMPLING_RES_THETA_H - 1);
}


// Lookup theta_diff index
// In:  [0 .. pi/2]
// Out: [0 .. 89]
static __host__ __device__ __inline__  unsigned int theta_diff_index(float theta_diff)
{
  theta_diff = optix::fmaxf(theta_diff, 0.0f);
  int idx = static_cast<int>(theta_diff*M_1_PIf*2.0f*BRDF_SAMPLING_RES_THETA_D);
  return optix::min(idx, BRDF_SAMPLING_RES_THETA_D - 1);
}


// Lookup phi_diff index
static __host__ __device__ __inline__  unsigned int phi_diff_index(float phi_diff)
{
  // Because of reciprocity, the BRDF is unchanged under
  // phi_diff -> phi_diff + M_PIf
  phi_diff = phi_diff < 0.0f ? phi_diff + M_PIf : phi_diff;
  phi_diff = optix::fmaxf(phi_diff, 0.0f);

  // In: phi_diff in [0 .. pi]
  // Out: tmp in [0 .. 179]
  int half_res = BRDF_SAMPLING_RES_PHI_D/2;
  int idx = static_cast<int>(phi_diff*M_1_PIf*half_res);
  return optix::min(idx, half_res - 1);
}


// Given a pair of incoming/outgoing angles, look up the BRDF.
#ifdef __CUDA_ARCH__
static __device__ __inline__ optix::float3 lookup_brdf_val(optix::buffer<float, 1> & brdf, float theta_half, float phi_half, float theta_diff, float phi_diff)
#else
static __host__ __device__ __inline__ optix::float3 lookup_brdf_val(float* brdf, float theta_half, float phi_half, float theta_diff, float phi_diff)
#endif
{
  // Find index.
  // Note that phi_half is ignored, since isotropic BRDFs are assumed
  int idx = phi_diff_index(phi_diff) +
    theta_diff_index(theta_diff)*BRDF_SAMPLING_RES_PHI_D/2 +
    theta_half_index(theta_half)*BRDF_SAMPLING_RES_PHI_D/2*BRDF_SAMPLING_RES_THETA_D;

  optix::float3 result;
  result.x = brdf[idx]*RED_SCALE;
  result.y = brdf[idx + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D/2]*GREEN_SCALE;
  result.z = brdf[idx + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D]*BLUE_SCALE;
  return optix::fmaxf(result, optix::make_float3(1e-6f));
}

#ifdef __CUDA_ARCH__
static __device__ __inline__ optix::float3 lookup_brdf_val(
  optix::buffer<float, 1> & brdf, const optix::float3& n, const optix::float3& normalized_wi, const optix::float3& normalized_wo)
#else
static __host__ __device__ __inline__ optix::float3 lookup_brdf_val(
  float* brdf, const optix::float3& n, const optix::float3& normalized_wi, const optix::float3& normalized_wo)
#endif
{
  optix::float3 t, b;
  create_onb(n, t, b);
  float a[9] = { t.x, t.y, t.z,
                 b.x, b.y, b.z,
                 n.x, n.y, n.z  };
  optix::Matrix3x3 m = optix::Matrix3x3(a);
  optix::float3 wi_t = m * normalized_wi;
  optix::float3 wo_t = m * normalized_wo;

  // Convert to halfangle / difference angle coordinates
  float theta_half, phi_half, theta_diff, phi_diff;
  vectors_to_half_diff_coords(wi_t, wo_t, theta_half, phi_half, theta_diff, phi_diff);

  optix::float3 result = lookup_brdf_val(brdf, theta_half, phi_half, theta_diff, phi_diff);
  return result;
}

#ifndef __CUDA_ARCH__
static __host__ __inline__ optix::float3 integrate_brdf(std::vector<float>& brdf, int N)

{
	optix::float3 sum = optix::make_float3(0.0f);

	srand(10);
	optix::float3 norm = optix::make_float3(0.0f, 0.0f, 1.0f);
	unsigned int seed = tea<16>(3, 17);

	optix::float3 last = optix::make_float3(0.0f);
	int i;
	for (i = 0; i < N; i++)
	{
		float zeta1 = rnd(seed);
		float zeta2 = rnd(seed);
		optix::float3 wi = sample_hemisphere_cosine(optix::make_float2(zeta1, zeta2), norm);
		float zeta3 = rnd(seed);
		float zeta4 = rnd(seed);
		optix::float3 wo = sample_hemisphere_cosine(optix::make_float2(zeta3, zeta4), norm);

		optix::float3 brdf_val = lookup_brdf_val(brdf.data(), norm, wi, wo);
		sum += brdf_val;
	}
	// There is a division by M_PIf that is compensated by the M_PIf in the numerator due to the cosine pdf.
	// Also, both cosines cancel out
	return M_PIf * sum / static_cast<float>(N);
}
#endif



#endif