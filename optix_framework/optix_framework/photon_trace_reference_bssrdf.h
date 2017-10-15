#pragma once
#include <device_common_data.h>
#include <random.h>
#include <sampling_helpers.h>
#include <environment_map.h>
#include "light.h"
#include "device_environment_map.h"
#include "optical_helper.h"
#include "phase_function.h"

#define RND_FUNC rnd_tea
#define EXTINCTION_DISTANCE_RR
//#define INCLUDE_SINGLE_SCATTERING

// Simple util to do plane ray intersection.
__forceinline__ __device__ bool intersect_plane(const optix::float3 & plane_origin, const optix::float3 & plane_normal, const optix::Ray & ray, float & intersection_distance)
{
	float denom = optix::dot(plane_normal, ray.direction);
	if (abs(denom) < 1e-6)
		return false; // Parallel: none or all points of the line lie in the plane.
	intersection_distance = optix::dot((plane_origin - ray.origin), plane_normal) / denom;
	return intersection_distance > ray.tmin && intersection_distance < ray.tmax;
}

__forceinline__ __device__ optix::float2 get_normalized_hemisphere_buffer_coordinates(float theta_o, float phi_o)
{
	const float phi_o_normalized = normalize_angle(phi_o) / (2.0f * M_PIf);
	const float theta_o_normalized = cosf(theta_o); //theta_o / (M_PIf * 0.5f);
	optix_assert(theta_o_normalized >= 0.0f);
	optix_assert(theta_o_normalized < 1.0f);
	optix_assert(phi_o_normalized < 1.0f);
	optix_assert(phi_o_normalized >= 0.0f);
	return optix::make_float2(phi_o_normalized, theta_o_normalized);
}

__forceinline__ __device__ optix::float2 get_normalized_hemisphere_buffer_angles(float theta_o_normalized, float phi_o_normalized)
{
	const float phi_o = phi_o_normalized * (2.0f * M_PIf);
	const float theta_o = acosf(theta_o_normalized); //theta_o / (M_PIf * 0.5f);
	return optix::make_float2(phi_o, theta_o);
}

__forceinline__ __device__ void store_values_in_buffer(const float cos_theta_o, const float phi_o, const float flux_E, BufPtr2D<float> & resulting_flux)
{
	const optix::size_t2 bins = resulting_flux.size();
	optix::float2 coords = get_normalized_hemisphere_buffer_coordinates(acosf(cos_theta_o), phi_o);
	optix::uint2 idxs = make_uint2(coords * make_float2(bins));
	optix_assert(flux_E >= 0.0f);
	optix_assert(!isnan(flux_E));

	// Atomic add to avoid thread conflicts
	if (!isnan(flux_E))
		atomicAdd(&resulting_flux[idxs], flux_E);
}

// Returns true if the photon has been absorbed, false otherwise
__forceinline__ __device__ bool scatter_photon(optix::float3& xp, optix::float3& wp, float & flux_t, BufPtr2D<float>& resulting_flux, const float3& xo, const float n2_over_n1, const float albedo, const float extinction, const float g, optix::uint & t, int starting_it, int executions)
{
	// Defining geometry
	const optix::float3 xi = optix::make_float3(0, 0, 0);
	const optix::float3 ni = optix::make_float3(0, 0, 1);
	const optix::float3 no = ni;
	const float n1_over_n2 = 1.0f / n2_over_n1;

	// We count executions to allow stop/resuming of this function.
	int i;
	for (i = starting_it; i < starting_it + executions; i++)
	{
		// If the flux is really small, we can stop here, the contribution is too small.
		if (flux_t < 1e-12)
			return true;

		const float rand = 1.0f - RND_FUNC(t); // RND_FUNC(t) in [0, 1). Avoids infinity when sampling exponential distribution.

		// Sampling new distance for photon, testing if it intersects the interface
		const float d = -log(rand) / extinction;
		optix::Ray ray = optix::make_Ray(xp, wp, 0, scene_epsilon, d);
		optix_assert(xp.z < 1e-6);
		optix_assert(xp.z > -INFINITY);

		float intersection_distance;
		if (!intersect_plane(xi, ni, ray, intersection_distance))
		{
			// We are still within the medium.
			// Russian roulette to check for absorption.
			float absorption_prob = RND_FUNC(t);
			if (absorption_prob > albedo)
			{
				optix_print("(%d) Absorption.\n", i);
				return true;
			}
			
			// We scatter now.
			xp = xp + wp * d;
			optix::float3 d_vec = xo - xp;
			const float d_vec_len = optix::length(d_vec);
			d_vec = d_vec / d_vec_len; // Normalizing

			optix_assert(optix::dot(d_vec, no) > 0.0f);

			const float cos_theta_21 = optix::max(optix::dot(d_vec, no), 0.0f); 
			const float cos_theta_21_sqr = cos_theta_21*cos_theta_21;

			// Note: we are checking the *exiting* ray, so we flip the relative ior 
			const float sin_theta_o_sqr = n2_over_n1*n2_over_n1*(1.0f - cos_theta_21_sqr);

			// If there is no total internal reflection, we accumulate
			if (sin_theta_o_sqr < 1.0f) 
			{
				const float cos_theta_o = sqrtf(1.0f - sin_theta_o_sqr);
				const float F_t = 1.0f - fresnel_R(cos_theta_21, n2_over_n1); 
				optix_assert(F_t < 1.0f);

				const float phi_21 = atan2f(d_vec.y, d_vec.x);		
				// The outgoing azimuthal angle is the same as the refracted vector, since the refracted vector 
				// w_12 points *towards* the surface, and the outgoing w_o points *away *from the surface.
				const float phi_o = phi_21;

#ifdef EXTINCTION_DISTANCE_RR
				if (RND_FUNC(t) < 1.0f - expf(-extinction*d_vec_len))
					return true;

				// The two exponentials cancel out when dividing out by the pdf, the extinction term remains.
				const float flux_E = flux_t * albedo * phase_HG(optix::dot(d_vec, wp), g)  * F_t / extinction; 
#else
				const float flux_E = flux_t * albedo * phase_HG(optix::dot(d_vec, wp), g) * expf(-extinction*d_vec_len) * F_t;
#endif
				// Not including single scattering, so i == 0 is not considered.

#ifdef INCLUDE_SINGLE_SCATTERING
				store_values_in_buffer(cos_theta_o, phi_o, flux_E, resulting_flux);
#else				
				if (i > 0)
				{
					store_values_in_buffer(cos_theta_o, phi_o, flux_E, resulting_flux);
				}
#endif
				optix_print("(%d) Scattering.  %f\n", i, flux_E);
			}
			// We choose a new direction sampling the phase function
			optix::float2 smpl = optix::make_float2(RND_FUNC(t), RND_FUNC(t));
			wp = optix::normalize(sample_HG(wp, g, smpl));
		}
		else
		{
			// We have intersected the plane. 
			const optix::float3 surface_point = xp + wp * intersection_distance;
			optix_assert(optix::dot(wp, no) > 0);

			// Calculate Fresnel coefficient
			const float cos_theta_p = optix::max(optix::dot(wp, no), 0.0f);
			const float F_r = fresnel_R(cos_theta_p, n2_over_n1); // assert F_t < 1

			// Reflect and turn to face inside.
			flux_t *= F_r;
			xp = surface_point;
			wp = reflect(wp, -no);
			optix_print("(%d) Reached surface %f.\n", i, F_r);
		}
	}
	return false;
}

#define REF_MAT_MILK 0
#define REF_MAT_WAX 1
#define REF_MAT_A 2
#define REF_MAT_B 3
#define REF_MAT_C 4
#define REF_MAT_D 5
#define REF_MAT_E 6
#define MATERIAL REF_MAT_E

__forceinline__ __device__ void get_default_material(float & theta_i, float & r, float & theta_s, float & albedo, float & extinction, float & g, float & n2_over_n1)
{
#if MATERIAL==REF_MAT_MILK
	theta_i = 20.0f;
	const float sigma_a = 0.0007f;
	const float sigma_s = 1.165;
	g = 0.7f;
	n2_over_n1 = 1.35f;
	r = 1.5f;
	theta_s = 0;
	albedo = sigma_s / (sigma_s + sigma_a);
	extinction = sigma_s + sigma_a;
#elif MATERIAL==REF_MAT_WAX
	theta_i = 20.0f;
	const float sigma_a = 0.5f;
	const float sigma_s = 1.0f;
	g = 0.0f;
	n2_over_n1 = 1.4f;
	r = 0.5f;
	theta_s = 90;
	albedo = sigma_s / (sigma_s + sigma_a);
	extinction = sigma_s + sigma_a;
#elif MATERIAL==REF_MAT_A
	theta_i = 30.0f;
	albedo = 0.6f;
	extinction = 1.0f;
	g = 0.0f;
	n2_over_n1 = 1.3f;
	r = 4.0f;
	theta_s = 0;
#elif MATERIAL==REF_MAT_B
	theta_i = 60.0f;
	theta_s = 60;
	r = 0.8f;
	albedo = 0.99f;
	extinction = 1.0f;
	g = -0.3f;
	n2_over_n1 = 1.4f;
#elif MATERIAL==REF_MAT_C
	theta_i = 70.0f;
	theta_s = 60;
	r = 1.0f;
	albedo = 0.3f;
	extinction = 1.0f;
	g = 0.9f;
	n2_over_n1 = 1.4f;
#elif MATERIAL==REF_MAT_D
	theta_i = 0.0f;
	theta_s = 105.0f;
	r = 4.0f;
	albedo = 0.5f;
	extinction = 1.0f;
	g = 0.0f;
	n2_over_n1 = 1.2f;
#elif MATERIAL==REF_MAT_E
	theta_i = 80.0f;
	theta_s = 165.0f;
	r = 2.0f;
	albedo = 0.8f;
	extinction = 1.0f;
	g = -0.3f;
	n2_over_n1 = 1.3f;
#endif
}