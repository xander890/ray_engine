#pragma once
#include "full_bssrdf_host_device_common.h"
#include <device_common_data.h>
#include <random.h>
#include <sampling_helpers.h>
#include <environment_map.h>
#include "light.h"
#include "device_environment_map.h"
#include "optical_helper.h"
#include "phase_function.h"
#include "empirical_bssrdf_utils.h"

#define RND_FUNC rnd_tea

// FIXME, we need to remove this or find a better way to express it without overhauling all the parameters.
rtDeclareVariable(float2, plane_size, , );

//#define EXTINCTION_DISTANCE_RR
//#define INCLUDE_SINGLE_SCATTERING

__forceinline__ __device__ void get_reference_scene_geometry(const float theta_i, const float r, const float theta_s, optix::float3 & xi, optix::float3 & wi, optix::float3 & ni, optix::float3 & xo, optix::float3 & no)
{
	wi = normalize(optix::make_float3(-sinf(theta_i), 0, cosf(theta_i)));
	// Geometry
	xi = optix::make_float3(0, 0, 0);
	ni = optix::make_float3(0, 0, 1);
	xo = xi + r * optix::make_float3(cos(theta_s), -sin(theta_s), 0);
	no = ni;
}

// Simple util to do plane ray intersection.
__forceinline__ __device__ bool intersect_plane(const optix::float3 & plane_origin, const optix::float3 & plane_normal, const optix::Ray & ray, float & intersection_distance)
{
	float denom = optix::dot(plane_normal, ray.direction);
	if (abs(denom) < 1e-6)
		return false; // Parallel: none or all points of the line lie in the plane.
	intersection_distance = optix::dot((plane_origin - ray.origin), plane_normal) / denom;
	return intersection_distance > ray.tmin && intersection_distance < ray.tmax;
}



__forceinline__ __device__ void store_values_in_buffer(const float theta_o, const float phi_o, const float flux_E, BufPtr2D<float> & resulting_flux)
{
	const optix::size_t2 bins = resulting_flux.size();
	optix::float2 coords = get_normalized_hemisphere_buffer_coordinates(theta_o, phi_o);
	optix::uint2 idxs = make_uint2(coords * make_float2(bins));
	optix_assert(flux_E >= 0.0f);
	optix_assert(!isnan(flux_E));

	// Atomic add to avoid thread conflicts
	if (!isnan(flux_E))
		atomicAdd(&resulting_flux[idxs], flux_E);
}

// Returns true if the photon has been absorbed, false otherwise
__forceinline__ __device__ bool scatter_photon_hemisphere(optix::float3& xp, optix::float3& wp, float & flux_t, BufPtr2D<float>& resulting_flux, const float3& xo, const float n2_over_n1, const float albedo, const float extinction, const float g, optix::uint & t, int starting_it, int executions)
{
	// Defining geometry
	const optix::float3 xi = optix::make_float3(0, 0, 0);
	const optix::float3 ni = optix::make_float3(0, 0, 1);
	const optix::float3 no = ni;

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


			// Note: we are checking the *exiting* ray, so we flip the relative ior 
			float3 wo;
			float cos_theta_o, cos_theta_21, F_r;
			// If there is no total internal reflection, we accumulate
			if (refract(-d_vec, -no, n2_over_n1, wo, F_r, cos_theta_21, cos_theta_o))
			{
				const float F_t = 1 - F_r;
								
				optix_assert(F_t < 1.0f);
				optix_assert(cos_theta_o >= 0);

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
					store_values_in_buffer(acosf(cos_theta_o), phi_o, flux_E, resulting_flux);
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

__forceinline__ __device__ optix::float2 get_normalized_planar_buffer_coordinates(const optix::float2 & coord)
{
	return (coord + plane_size/2) / plane_size;
}

__forceinline__ __device__ optix::float2 get_planar_buffer_coordinates(const optix::float2 & normalized_coord)
{
	return (normalized_coord * plane_size) - plane_size/2;
}


__forceinline__ __device__ void store_values_in_buffer_planar(const optix::float2 & xo_plane, const float flux_E, BufPtr2D<float> & resulting_flux)
{
	const optix::size_t2 bins = resulting_flux.size();
	optix::float2 coords = get_normalized_planar_buffer_coordinates(xo_plane);

	// Store only if on the plane
	if (coords.x >= 0.0f && coords.x < 1.0f && coords.y >= 0.0f && coords.y < 1.0f)
	{
		optix::uint2 idxs = make_uint2(coords * make_float2(bins));
		optix_assert(flux_E >= 0.0f);
		optix_assert(!isnan(flux_E));

		// Atomic add to avoid thread conflicts
		if (!isnan(flux_E))
			atomicAdd(&resulting_flux[idxs], flux_E);
	}
}


// Returns true if the photon has been absorbed, false otherwise
__forceinline__ __device__ bool scatter_photon_planar(optix::float3& xp, optix::float3& wp, float & flux_t, BufPtr2D<float>& resulting_flux, const float3& xo, const float n2_over_n1, const float albedo, const float extinction, const float g, optix::uint & t, int starting_it, int executions)
{
	// Defining geometry
	const optix::float3 xi = optix::make_float3(0, 0, 0);
	const optix::float3 ni = optix::make_float3(0, 0, 1);
	const optix::float3 no = ni;

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

			float outgoing_flux = (1.0f - F_r) * flux_t;
			store_values_in_buffer_planar(make_float2(surface_point.x, surface_point.y), outgoing_flux, resulting_flux);

			// Reflect and turn to face inside.
			flux_t *= F_r;
			xp = surface_point;
			wp = reflect(wp, -no);
			optix_print("(%d) Reached surface %f.\n", i, outgoing_flux);
		}
	}
	return false;
}

__forceinline__ __device__ bool scatter_photon(int mode, optix::float3& xp, optix::float3& wp, float & flux_t, BufPtr2D<float>& resulting_flux, const float3& xo, const float n2_over_n1, const float albedo, const float extinction, const float g, optix::uint & t, int starting_it, int executions)
{
	return (mode == BSSRDF_OUTPUT_HEMISPHERE) ? scatter_photon_hemisphere(xp, wp, flux_t, resulting_flux, xo, n2_over_n1, albedo, extinction, g, t, starting_it, executions) :
		scatter_photon_planar(xp, wp, flux_t, resulting_flux, xo, n2_over_n1, albedo, extinction, g, t, starting_it, executions);
}
