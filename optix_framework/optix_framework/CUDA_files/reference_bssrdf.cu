// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011



#include <device_common_data.h>
#include <random.h>
#include <sampling_helpers.h>
#include <environment_map.h>
#include "light.h"
#include "device_environment_map.h"
#include "optical_helper.h"
#include "phase_function.h"

using namespace optix;
// Window variables
rtBuffer<float,2> resulting_flux;

__forceinline__ __device__ bool intersect_plane(const optix::float3 & plane_origin, const optix::float3 & plane_normal, const optix::Ray & ray, float & intersection_distance)
{
	float denom = optix::dot(plane_normal, ray.direction);
	if (abs(denom) < 1e-6)
		return false; // Parallel: none or all points of the line lie in the plane.
	intersection_distance = optix::dot((plane_origin - ray.origin), plane_normal) / denom;
	return intersection_distance > ray.tmin && intersection_distance < ray.tmax;
}

// Assumes an infinite plane with normal (0,0,1)
__forceinline__ __device__ void infinite_plane_scatter_searchlight(const optix::float3& wi, const float incident_power, const float r, const float theta_s, const float n1_over_n2, const float sigma_a, const float sigma_s, const float g, optix::uint & t)
{
	const int theta_bins = resulting_flux.size().y;
	const int phi_bins = resulting_flux.size().x;
	
	// Geometry
	const optix::float3 xi = optix::make_float3(0, 0, 0);
	const optix::float3 ni = optix::make_float3(0, 0, 1);
	const optix::float3 xo = xi + r * optix::make_float3(cos(theta_s), sin(theta_s), 0);
	const optix::float3 no = ni;

	// Optical properties
	const float albedo = sigma_s / (sigma_s + sigma_a);
	const float extinction = sigma_s + sigma_a;
	const float critical_angle = asinf(n1_over_n2);
	const float n2_over_n1 = 1.0f / n1_over_n2;

	// Refraction
	const float cos_theta_i = optix::max(optix::dot(wi, ni), 0.0f);
	const float cos_theta_i_sqr = cos_theta_i*cos_theta_i;
	const float sin_theta_t_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_i_sqr);
	const float cos_theta_t_i = sqrt(1.0f - sin_theta_t_sqr);
	const optix::float3 w12 = n1_over_n2*(cos_theta_i*ni - wi) - ni*cos_theta_t_i;

	float flux_t = incident_power;
	optix::float3 xp = xi;
	optix::float3 wp = w12;
	while (true)
	{
		float rand = rnd(t);
		float d = -log(rand) / extinction;
		float intersection_distance;
		optix::Ray ray = optix::make_Ray(xp, wp, 0, scene_epsilon, d);
		if (!intersect_plane(xi, ni, ray, intersection_distance))
		{
			// Still within the medium...
			xp = xp + wp * d;
			optix::float3 d_vec = xo - xp;
			const float d_vec_len = optix::length(d_vec);
			d_vec = d_vec / d_vec_len; // Normalizing

			const float cos_theta_21 = optix::max(optix::dot(d_vec, no), 0.0f); // This should be positive
			const float cos_theta_21_sqr = cos_theta_21*cos_theta_21;

			// Note: we flip the relative ior because we are going outside from inside
			const float sin_theta_o_sqr = n2_over_n1*n2_over_n1*(1.0f - cos_theta_21_sqr);

			if (sin_theta_o_sqr >= 1.0f) // Total internal reflection, no accumulation
			{
				continue;
			}
			else
			{
				const float cos_theta_o = sqrt(1.0f - sin_theta_o_sqr);
				const float F_t = 1.0f - fresnel_R(cos_theta_21, n2_over_n1); // assert F_t < 1
				const float phi_21 = atan2f(d_vec.y, d_vec.x);

				// Store atomically in appropriate spot.
				const float phi_o = phi_21 + M_PIf > 2.0f * M_PIf ? phi_21 - M_PIf : phi_21 + M_PIf;
				const float theta_o = acosf(cos_theta_o);
				float flux_E = flux_t * albedo * eval_HG(optix::dot(d_vec, wp), g) * expf(-extinction*d_vec_len)*F_t;
				int bin_theta_o = int(theta_o * M_1_PIf * theta_bins);
				int bin_phi_o = int(phi_o * 0.5f * M_1_PIf * phi_bins);
				atomicAdd(&resulting_flux[make_uint2(bin_phi_o, bin_theta_o)], flux_E);

			}

			if (rnd(t) > albedo)
				break;

			// We scatter, so we choose a new direction sampling the phase function
			wp = sample_HG(wp, g, t);
		}
		else
		{
			// We are going out!
			const optix::float3 surface_point = xp + wp * intersection_distance;
			const float cos_theta_p = optix::max(optix::dot(wp, no), 0.0f); // This should be positive
			const float cos_theta_p_sqr = cos_theta_p*cos_theta_p;

			const float sin_theta_o_sqr = n2_over_n1*n2_over_n1*(1.0f - cos_theta_p_sqr);

			if (sin_theta_o_sqr >= 1.0f) // Total internal reflection,
			{
				wp = -reflect(wp, no); // Reflect and turn to face inside.
				const float cos_theta_o = sqrt(1.0f - sin_theta_o_sqr);
				const float F_t = 1.0f - fresnel_R(cos_theta_o, n1_over_n2); // assert F_t < 1
				flux_t *= F_t;
			}
			else
			{
				// We are outside
				break;
			}
		}
	}
}

#define MILK 0
#define WAX 1
#define MATERIAL MILK

RT_PROGRAM void reference_bssrdf_camera()
{
	uint idx = launch_index.x;
	const float theta_i = 20.0f;
	optix::uint t = tea<16>(idx, frame);

	const float incident_power = 1.0f;

#if MATERIAL==MILK
	const float sigma_a = 0.0007f;
	const float sigma_s = 1.165;
	const float g = 0.7f;
	const float n1_over_n2 = 1.0f / 1.35f;
	const float r = 1.5f;
	const float theta_s = 0;
#else
	const float sigma_a = 0.5f;
	const float sigma_s = 1.0f;
	const float g = 0.0f;
	const float n1_over_n2 = 1.0f / 1.4f;
	const float r = 0.5f;
	const float theta_s = 90;
#endif
	const float theta_i_rad = deg2rad(theta_i);
	const float theta_s_rad = deg2rad(theta_s);
	const optix::float3 wi = optix::make_float3(sinf(theta_i_rad), 0, cosf(theta_i_rad));
	infinite_plane_scatter_searchlight(wi, incident_power, r, theta_s, n1_over_n2, sigma_a, sigma_s, g, t);
	
}

