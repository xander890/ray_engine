#pragma once
#include "full_bssrdf_host_device_common.h"
#include <device_common_data.h>
#include "photon_trace_structs.h"
#include <random.h>
#include <sampling_helpers.h>
#include <environment_map.h>
#include "light.h"
#include "device_environment_map.h"
#include "optical_helper.h"
#include "phase_function.h"
#include "empirical_bssrdf_utils.h"

// FIXME, we need to remove this or find a better way to express it without overhauling all the parameters.
rtDeclareVariable(optix::float2, plane_size, , );

__forceinline__ __device__ void get_reference_scene_geometry(const float theta_i, const float r, const float theta_s, optix::float3 & xi, optix::float3 & wi, optix::float3 & ni, optix::float3 & xo, optix::float3 & no)
{
	wi = normalize(optix::make_float3(-sinf(theta_i), 0, cosf(theta_i)));
	// Geometry
	xi = optix::make_float3(0, 0, 0);
	ni = optix::make_float3(0, 0, 1);
	xo = xi + r * optix::make_float3(cos(theta_s), sin(theta_s), 0);
	no = ni;
}

// Simple util to do plane ray intersection.
__forceinline__ __device__ bool intersect_plane(const optix::float3 & plane_origin, const optix::float3 & plane_normal, const optix::Ray & ray, float & intersection_distance)
{
	float denom = optix::dot(plane_normal, ray.direction);
	if (abs(denom) < 1e-12)
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
	optix_print("Storing flux %f\n", flux_E);

	// Atomic add to avoid thread conflicts
	if (!isnan(flux_E))
		atomicAdd(&resulting_flux[idxs], flux_E);
}


// Returns true if the photon has been absorbed, false otherwise
__forceinline__ __device__ bool scatter_photon_hemisphere_mcml(const BSSRDFRendererData & geometry_data, optix::float3& xp, optix::float3& wp, float & flux_t, BufPtr2D<float>& resulting_flux, const float n2_over_n1, const float albedo, const float extinction, const float g, SEED_TYPE & t, int starting_it, int executions)
{
	// Defining geometry
	const optix::float3 xi = optix::make_float3(0, 0, 0);
	const optix::float3 ni = optix::make_float3(0, 0, 1);
	const optix::float3 no = ni;

	// We count executions to allow stop/resuming of this function.
	int i;
	for (i = starting_it; i < starting_it + executions; i++)
	{
#ifdef TERMINATE_ON_SMALL_FLUX
		// If the flux is really small, we can stop here, the contribution is too small.
		if (flux_t < 1e-12)
			return true;
#endif
		const float rand = RND_FUNC(t);

		// Sampling new distance for photon, testing if it intersects the interface
		const float d = -log(rand) / extinction;
		optix::Ray ray = optix::make_Ray(xp, wp, 0, 1e-12f, d);

        if(!isfinite(xp.z))
        {
            return true;
        }

        optix_assert(xp.z <= 1e-6);
		optix_assert(xp.z > -INFINITY);

        optix_print("%d (launch %d) -%f - xp %f %f %f wp %f %f %f \n", i, launch_index.x,optix::dot(wp, no), xp.x, xp.y, xp.z, wp.x, wp.y, wp.z);

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
            xp.z = fminf(xp.z, 0);
		}
		else
		{
			// We have intersected the plane.
			const optix::float3 surface_point = xp + wp * intersection_distance;
			optix_assert(optix::dot(wp, no) > 0);

			// Calculate refracted vector.
            float F_r, cos_theta_i, cos_theta_t;
            optix::float3 wo;
            refract(-wp, -no, n2_over_n1, wo, F_r, cos_theta_i, cos_theta_t);

			const float reflection_probability = RND_FUNC(t);

			if(reflection_probability < F_r)
			{
                optix_print("(%d) Internal reflection. %f %f %f\n", i, no.x, no.y, no.z);

                // Reflect and turn to face inside.
				xp = surface_point;
                xp.z = 0; // To avoid numerical imprecisions.
				wp = reflect(wp, -no);
			}
			else
			{

                // Photons escapes the medium, we store it.
				// Check if we are in the correct spatial bin!
                optix::float3 diff = surface_point - xi;
                float r = optix::length(diff);
                float theta_s = atan2(diff.y, diff.x);
                optix_assert(r >= 0);

                bool is_r_in_bin = r >= geometry_data.mRadius.x && r < geometry_data.mRadius.y;
                bool is_theta_s_in_bin = theta_s >= geometry_data.mThetas.x && theta_s < geometry_data.mThetas.y;
                bool is_p_in_bin = is_r_in_bin && is_theta_s_in_bin;

				float phi_o = atan2(wo.y, wo.x);
				float theta_o = acosf(wo.z);

                float flux_to_store = flux_t * 1.0f / (geometry_data.mArea * geometry_data.mSolidAngle * dot(wo, no));

                if(is_p_in_bin && i > 1) // No single scattering.
                {
                    optix_print("(%d) Refraction. theta_o %f phi_o %f - %f\n", i, theta_o, phi_o, flux_to_store);

                    store_values_in_buffer(theta_o, phi_o, flux_to_store, resulting_flux);
                }
                // We are done with this random walk.
                return true;
			}
		}
	}
	return false;
}

// Returns true if the photon has been absorbed, false otherwise
__forceinline__ __device__ bool scatter_photon_hemisphere_connections_correct(const BSSRDFRendererData & geometry_data, optix::float3& xp, optix::float3& wp, float & flux_t, BufPtr2D<float>& resulting_flux, const float n2_over_n1, const float albedo, const float extinction, const float g, SEED_TYPE & t, int starting_it, int executions)
{
    // Defining geometry
    const optix::float3 xi = optix::make_float3(0, 0, 0);
    const optix::float3 ni = optix::make_float3(0, 0, 1);
    const optix::float3 no = ni;

    // We count executions to allow stop/resuming of this function.
    int i;
    for (i = starting_it; i < starting_it + executions; i++)
    {
#ifdef TERMINATE_ON_SMALL_FLUX
        // If the flux is really small, we can stop here, the contribution is too small.
        if (flux_t < 1e-12)
            return true;
#endif
        const float rand = RND_FUNC(t); // RND_FUNC(t) in [0, 1). Avoids infinity when sampling exponential distribution.

        // Sampling new distance for photon, testing if it intersects the interface
        const float d = -log(rand) / extinction;
        optix::Ray ray = optix::make_Ray(xp, wp, 0, 1e-12, d);

        if(!isfinite(xp.z))
        {
            return true;
        }

        optix_assert(xp.z <= 1e-6);
        optix_assert(xp.z > -INFINITY);

        float intersection_distance;
        if (!intersect_plane(xi, ni, ray, intersection_distance))
        {
            // We scatter now.
            xp = xp + wp * d;
            xp.z = fminf(xp.z, 0);  // This is needed to eliminate edge cases that give xp.z > 0 due to numerical precision.

            float r = geometry_data.mRadius.x + geometry_data.mDeltaR * RND_FUNC(t);
            float theta_s = geometry_data.mThetas.x + geometry_data.mDeltaThetas * RND_FUNC(t);

			const optix::float3 xo = xi + r * optix::make_float3(cos(theta_s), sin(theta_s), 0);
            optix::float3 w21 = xo - xp;
            const float xoxp = optix::length(w21);
            w21 = w21 / xoxp; // Normalizing

            optix_assert(optix::dot(w21, no) >= 0.0f);

            // Note: we are checking the *exiting* ray, so we flip the relative ior
            float3 wo;
            float cos_theta_o, cos_theta_21, R21;
            // If there is no total internal reflection, we accumulate
            if (refract(-w21, -no, n2_over_n1, wo, R21, cos_theta_21, cos_theta_o))
            {
                const float T21 = 1 - R21;
                optix_assert(R21 <= 1.0f);
                optix_assert(cos_theta_o >= 0);

                const float phi_21 = atan2f(w21.y, w21.x);
                // The outgoing azimuthal angle is the same as the refracted vector, since the refracted vector
                // w_12 points *towards* the surface, and the outgoing w_o points *away *from the surface.
                const float phi_o = phi_21;

				float geometry_term = fabsf(optix::dot(w21, no)) / (xoxp*xoxp);
                float bssrdf_E = albedo * flux_t * phase_HG(optix::dot(wp, w21), g) * (T21 / dot(wo, no)) * geometry_term * exp(-extinction * xoxp);
				bssrdf_E *= r * (2.0f / (geometry_data.mRadius.x + geometry_data.mRadius.y)) * (1.0f / geometry_data.mSolidAngle);

                optix_print("flux_t %f, albedo %f, p %f, exp %f, F %f\n",  flux_t, albedo , phase_HG(optix::dot(w21, wp), g) , expf(-extinction*xoxp) , T21);

                // Not including single scattering, so i == 0 is not considered.
                if (i > 0)
                {
                    store_values_in_buffer(acosf(cos_theta_o), phi_o, bssrdf_E, resulting_flux);
                }

                optix_print("(%d) Scattering.  %f\n", i, bssrdf_E);
            }
            // We choose a new direction sampling the phase function
            optix::float2 smpl = optix::make_float2(RND_FUNC(t), RND_FUNC(t));
            wp = optix::normalize(sample_HG(wp, g, smpl));

            // We are still within the medium.
            // Russian roulette to check for absorption.
            float absorption_prob = RND_FUNC(t);
            if (absorption_prob > albedo)
            {
                optix_print("(%d) Absorption.\n", i);
                return true;
            }
        }
        else
        {
            // We have intersected the plane.
            const optix::float3 surface_point = xp + wp * intersection_distance;
                    optix_assert(optix::dot(wp, no) > 0);

            float3 wo; float F_r;
            refract(-wp, -no, n2_over_n1, wo, F_r);

            // Reflect and turn to face inside.
            flux_t *= F_r;
            xp = surface_point;
            wp = reflect(wp, -no);
            xp.z = 0; // This is needed to eliminate edge cases that give xp.z > 0 due to numerical precision.
            optix_print("(%d) Reached surface %f.\n", i, F_r);
        }
    }
    return false;
}


// Returns true if the photon has been absorbed, false otherwise
__forceinline__ __device__ bool scatter_photon_hemisphere_connections_correct_bias_compensation(const BSSRDFRendererData & geometry_data, optix::float3& xp, optix::float3& wp, float & flux_t, BufPtr2D<float>& resulting_flux, const float n2_over_n1, const float albedo, const float extinction, const float g, SEED_TYPE & t, int starting_it, int executions, float bias)
{
    // Defining geometry
    const optix::float3 xi = optix::make_float3(0, 0, 0);
    const optix::float3 ni = optix::make_float3(0, 0, 1);
    const optix::float3 no = ni;


    // We count executions to allow stop/resuming of this function.
    int i;
    for (i = starting_it; i < starting_it + executions; i++)
    {
#ifdef TERMINATE_ON_SMALL_FLUX
        // If the flux is really small, we can stop here, the contribution is too small.
        if (flux_t < 1e-12)
            return true;
#endif
        const float rand = RND_FUNC(t); // RND_FUNC(t) in [0, 1). Avoids infinity when sampling exponential distribution.

        // Sampling new distance for photon, testing if it intersects the interface

        const float d = -log(rand) / extinction;

        optix::Ray ray = optix::make_Ray(xp, wp, 0, 1e-12, d);

        if (!isfinite(xp.z)) {
            return true;
        }

        optix_assert(xp.z <= 1e-6);
        optix_assert(xp.z > -INFINITY);

        float intersection_distance;
        if (!intersect_plane(xi, ni, ray, intersection_distance)) {
            // We scatter now.
            xp = xp + wp * d;
            xp.z = fminf(xp.z, 0);  // This is needed to eliminate edge cases that give xp.z > 0 due to numerical precision.

            float r = geometry_data.mRadius.x + geometry_data.mDeltaR * RND_FUNC(t);
            float theta_s = geometry_data.mThetas.x + geometry_data.mDeltaThetas * RND_FUNC(t);

            const optix::float3 xo = xi + r * optix::make_float3(cos(theta_s), sin(theta_s), 0);
            optix::float3 w21 = xo - xp;
            const float xoxp = optix::length(w21);
            w21 = w21 / xoxp; // Normalizing

            optix_assert(optix::dot(w21, no) >= 0.0f);

            // Note: we are checking the *exiting* ray, so we flip the relative ior
            float3 wo;
            float cos_theta_o, cos_theta_21, R21;
            // If there is no total internal reflection, we accumulate
            if (refract(-w21, -no, n2_over_n1, wo, R21, cos_theta_21, cos_theta_o)) {
                const float T21 = 1 - R21;
                        optix_assert(R21 <= 1.0f);
                        optix_assert(cos_theta_o >= 0);

                const float phi_21 = atan2f(w21.y, w21.x);
                // The outgoing azimuthal angle is the same as the refracted vector, since the refracted vector
                // w_12 points *towards* the surface, and the outgoing w_o points *away *from the surface.
                const float phi_o = phi_21;


                const float mfp_bias = 1 / bias;
                const float b = bias*bias;

                float G = 1.0f / (xoxp * xoxp);
                float geometry_term = min(b, G);

                float bssrdf_E = albedo * flux_t * phase_HG(optix::dot(wp, w21), g) * fabsf(optix::dot(w21, no)) * (T21 / dot(wo, no));
                bssrdf_E *= r * (2.0f / (geometry_data.mRadius.x + geometry_data.mRadius.y)) * (1.0f / geometry_data.mSolidAngle);

                optix_print("flux_t %f, albedo %f, p %f, exp %f, F %f\n", flux_t, albedo,
                            phase_HG(optix::dot(w21, wp), g), expf(-extinction * xoxp), T21);


                optix_print("(%d) Scattering.  %f\n", i, bssrdf_E);

                float bssrdf_to_store = bssrdf_E * geometry_term * expf(-extinction * xoxp);

#if 1
                // Not including single scattering, so i == 0 is not considered.
                if(i > 0 && xoxp < mfp_bias)
                {
                    float G_prime = 1 - b * xoxp * xoxp;
                    optix_assert(G_prime > 0);
                    optix_assert(fabsf(geometry_term - b) < 1e-20f);

                    bssrdf_to_store += bssrdf_E * expf(-extinction * xoxp) * G_prime;
                    float sign_dir = RND_FUNC(t) > 0.5f? -1.0f : 1.0f;
                    optix::float3 old_dir = wp;
                    optix::float3 new_dir = sign_dir * w21;
                    optix::float3 new_pos = xp;
                    while(true)
                    {
                        float rand_bias = RND_FUNC(t);
                        float d_bias = -log(rand_bias) / extinction;

                        new_pos = new_pos + new_dir * d_bias;
                        float distance_from_xo = optix::length(new_pos - xo);

                        // If we go outside or too far.
                        if(d_bias > mfp_bias)
                            break;

                        if(new_pos.z >= 0.0f || distance_from_xo > mfp_bias)
                            break;

                        G_prime *= (1 - b * d_bias * d_bias);
                        bssrdf_to_store += G_prime * bssrdf_E * exp(-extinction * distance_from_xo) * phase_HG(dot(old_dir, new_dir), g) * 2.0f;

                        old_dir = new_dir;
                        float sign_dir = RND_FUNC(t) > 0.5f? -1.0f : 1.0f;
                        new_dir = sign_dir * w21;
                    }
                }
#else
                if(i > 0 && xoxp < mfp_bias)
                {
                    float G = (1.0f - b*xoxp*xoxp);
                    bssrdf_to_store += bssrdf_E * G;
                    optix::float3 new_xp = xp;
                    optix::float3 new_wp = -w21;

                    while(false)
                    {
                        float rand_bias = RND_FUNC(t); // RND_FUNC(t) in [0, 1). Avoids infinity when sampling exponential distribution.
                        float d_bias = -log(rand_bias) / extinction;

                        if(d_bias >= mfp_bias) {
                            optix_print("Bias computation terminated. \n");
                            break;
                        }

                        optix::float2 smpl = optix::make_float2(RND_FUNC(t), RND_FUNC(t));
                        new_wp = sample_HG(new_wp, g, smpl);
                        optix::Ray ray_bias = optix::make_Ray(new_xp, new_wp, 0, 1e-12, d_bias);
                        float intersection_distance_bias;
                        if(!intersect_plane(xi, ni, ray_bias, intersection_distance_bias))
                        {
                            new_xp = new_xp + new_wp * d_bias;
                            new_xp.z = fminf(new_xp.z, 0);
                            optix::float3 new_w21 = xo - new_xp;
                            const float new_xoxp = optix::length(new_w21);
                            new_w21 = new_w21 / new_xoxp; // Normalizing
                            float3 new_wo;
                            float new_R21;
                            // If there is no total internal reflection, we accumulate
                            if (refract(-new_w21, -no, n2_over_n1, new_wo, new_R21)) {
                                const float new_T21 = 1 - new_R21;
                                optix_assert(new_R21 <= 1.0f);

                                // The outgoing azimuthal angle is the same as the refracted vector, since the refracted vector
                                // w_12 points *towards* the surface, and the outgoing w_o points *away *from the surface.
                                float new_geometry_term = min(bound, 1.0f / (new_xoxp * new_xoxp));
                                new_geometry_term *= fabsf(optix::dot(new_w21, no));
                                float new_bssrdf_E = albedo * flux_t * phase_HG(optix::dot(new_wp, new_w21), g) * (new_T21 / dot(new_wo, no)) * new_geometry_term *
                                        expf(-extinction * new_xoxp);
                                new_bssrdf_E *= r * (2.0f / (geometry_data.mRadius.x + geometry_data.mRadius.y)) *
                                            (1.0f / geometry_data.mSolidAngle);

                                optix_print("Add... %d %f %f\n", launch_index.x,  G, new_bssrdf_E);
                                G *= (1 - bound * d_bias * d_bias);
                                // Not including single scattering, so i == 0 is not considered.
                                bssrdf_to_store += new_bssrdf_E * G;

                            }
                            else break;
                        }
                        else break;
                    }
                }
#endif

                if (i > 0) {
                    store_values_in_buffer(acosf(cos_theta_o), phi_o, bssrdf_to_store, resulting_flux);
                }

            }
            // We choose a new direction sampling the phase function
            optix::float2 smpl = optix::make_float2(RND_FUNC(t), RND_FUNC(t));
            wp = optix::normalize(sample_HG(wp, g, smpl));

            // We are still within the medium.
            // Russian roulette to check for absorption.
            float absorption_prob = RND_FUNC(t);
            if (absorption_prob > albedo) {
                optix_print("(%d) Absorption.\n", i);
                return true;
            }

        } else {
            // We have intersected the plane.
            const optix::float3 surface_point = xp + wp * intersection_distance;
                    optix_assert(optix::dot(wp, no) > 0);

            float3 wo;
            float F_r;
            refract(-wp, -no, n2_over_n1, wo, F_r);

            // Reflect and turn to face inside.
            flux_t *= F_r;
            xp = surface_point;
            wp = reflect(wp, -no);
            xp.z = 0; // This is needed to eliminate edge cases that give xp.z > 0 due to numerical precision.
            optix_print("(%d) Reached surface %f.\n", i, F_r);
        }

    }
    return false;
}

// Returns true if the photon has been absorbed, false otherwise
__forceinline__ __device__ bool scatter_photon_hemisphere_connections(const BSSRDFRendererData & geometry_data, optix::float3& xp, optix::float3& wp, float & flux_t, BufPtr2D<float>& resulting_flux, const float3& xo, const float n2_over_n1, const float albedo, const float extinction, const float g, SEED_TYPE & t, int starting_it, int executions)
{
	const float scattering = albedo * extinction;
	// Defining geometry
	const optix::float3 xi = optix::make_float3(0, 0, 0);
	const optix::float3 ni = optix::make_float3(0, 0, 1);
	const optix::float3 no = ni;

	// We count executions to allow stop/resuming of this function.
	int i;
	for (i = starting_it; i < starting_it + executions; i++)
	{
#ifdef TERMINATE_ON_SMALL_FLUX
		// If the flux is really small, we can stop here, the contribution is too small.
		if (flux_t < 1e-12)
			return true;
#endif
		const float rand = 1.0f - RND_FUNC(t); // RND_FUNC(t) in [0, 1). Avoids infinity when sampling exponential distribution.

		// Sampling new distance for photon, testing if it intersects the interface
		const float d = -log(rand) / extinction;
		optix::Ray ray = optix::make_Ray(xp, wp, 0, 1e-12, d);

        if(!isfinite(xp.z))
        {
            return true;
        }

        optix_assert(xp.z <= 1e-6);
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
            xp.z = fminf(xp.z, 0);

			optix_assert(optix::dot(d_vec, no) > 0.0f);

			// Note: we are checking the *exiting* ray, so we flip the relative ior 
			float3 wo;
			float cos_theta_o, cos_theta_21, F_r;
			// If there is no total internal reflection, we accumulate
			if (refract(-d_vec, -no, n2_over_n1, wo, F_r, cos_theta_21, cos_theta_o))
			{
				const float F_t = 1 - F_r;
								
				optix_assert(F_t <= 1.0f);
				optix_assert(cos_theta_o >= 0);

				const float phi_21 = atan2f(d_vec.y, d_vec.x);		
				// The outgoing azimuthal angle is the same as the refracted vector, since the refracted vector 
				// w_12 points *towards* the surface, and the outgoing w_o points *away *from the surface.
				const float phi_o = phi_21;

				float flux_E = flux_t * scattering * phase_HG(optix::dot(d_vec, wp), g) * expf(-extinction*d_vec_len) * F_t;
#ifdef INCLUDE_GEOMETRIC_TERM
				optix_print("Geometric.\n");
				optix_assert(fabsf(optix::dot(d_vec, no) - cos_theta_21) < 1e-4f);
				flux_E *= fabsf(cos_theta_21) / (d_vec_len*d_vec_len);
#endif 
				optix_print("flux_t %f, albedo %f, p %f, exp %f, F %f\n",  flux_t, albedo , phase_HG(optix::dot(d_vec, wp), g) , expf(-extinction*d_vec_len) , F_t);

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

            float3 wo; float F_r;
            refract(-wp, -no, n2_over_n1, wo, F_r);

			// Reflect and turn to face inside.
			flux_t *= F_r;
			xp = surface_point;
			wp = reflect(wp, -no);
            xp.z = 0;
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
__forceinline__ __device__ bool scatter_photon_planar(optix::float3& xp, optix::float3& wp, float & flux_t, BufPtr2D<float>& resulting_flux, const float3& xo, const float n2_over_n1, const float albedo, const float extinction, const float g, SEED_TYPE & t, int starting_it, int executions)
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
		//if (flux_t < 1e-12)
		//	return true;

		const float rand = 1.0f - RND_FUNC(t); // RND_FUNC(t) in [0, 1). Avoids infinity when sampling exponential distribution.

											   // Sampling new distance for photon, testing if it intersects the interface
		const float d = -log(rand) / extinction;
		optix::Ray ray = optix::make_Ray(xp, wp, 0, 0, d);
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
			store_values_in_buffer_planar(optix::make_float2(surface_point.x, surface_point.y), outgoing_flux, resulting_flux);

			// Reflect and turn to face inside.
			flux_t *= F_r;
			xp = surface_point;
			wp = reflect(wp, -no);
			optix_print("(%d) Reached surface %f.\n", i, outgoing_flux);
		}
	}
	return false;
}

// Returns true if the photon has been absorbed, false otherwise
__forceinline__ __device__ bool scatter_photon_hemisphere(IntegrationMethod::Type integration, const BSSRDFRendererData & geometry_data, optix::float3& xp, optix::float3& wp, float & flux_t, BufPtr2D<float>& resulting_flux, const float3& xo, const float n2_over_n1, const float albedo, const float extinction, const float g, SEED_TYPE & t, int starting_it, int executions, float bias)
{
	if(integration == IntegrationMethod::MCML)
		return scatter_photon_hemisphere_mcml(geometry_data, xp, wp, flux_t, resulting_flux, n2_over_n1, albedo, extinction, g, t, starting_it, executions);
	else if (integration == IntegrationMethod::CONNECTIONS)
		return scatter_photon_hemisphere_connections(geometry_data, xp, wp, flux_t, resulting_flux, xo, n2_over_n1, albedo, extinction, g, t, starting_it, executions);
    else if (integration == IntegrationMethod::CONNECTIONS_WITH_FIX)
        return scatter_photon_hemisphere_connections_correct(geometry_data, xp, wp, flux_t, resulting_flux, n2_over_n1, albedo, extinction, g, t, starting_it, executions);
    else if (integration == IntegrationMethod::CONNECTIONS_WITH_BIAS_REDUCTION)
        return scatter_photon_hemisphere_connections_correct_bias_compensation(geometry_data, xp, wp, flux_t, resulting_flux, n2_over_n1, albedo, extinction, g, t, starting_it, executions, bias);
    else
		return false;
}


__forceinline__ __device__ bool scatter_photon(OutputShape::Type shape, IntegrationMethod::Type integration, const BSSRDFRendererData & geometry_data, optix::float3& xp, optix::float3& wp, float & flux_t, BufPtr2D<float>& resulting_flux, const float3& xo, const float n2_over_n1, const float albedo, const float extinction, const float g, SEED_TYPE & t, int starting_it, int executions, float bias = 0)
{
	if(shape == OutputShape::HEMISPHERE)
        return scatter_photon_hemisphere(integration, geometry_data, xp, wp, flux_t, resulting_flux, xo, n2_over_n1, albedo, extinction, g, t, starting_it, executions, bias);
    else
		return scatter_photon_planar(xp, wp, flux_t, resulting_flux, xo, n2_over_n1, albedo, extinction, g, t, starting_it, executions);
}
