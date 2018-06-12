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
#include "empirical_bssrdf_common.h"

// FIXME, we need to remove this or find a better way to express it without overhauling all the parameters.
rtDeclareVariable(optix::float2, plane_size, ,);

_fn void get_reference_scene_geometry(const float theta_i, const float r, const float theta_s, optix::float3 &xi,
        optix::float3 &wi, optix::float3 &ni, optix::float3 &xo, optix::float3 &no)
{
    wi = normalize(optix::make_float3(-sinf(theta_i), 0, cosf(theta_i)));
    // MeshGeometry
    xi = optix::make_float3(0, 0, 0);
    ni = optix::make_float3(0, 0, 1);
    xo = xi + r * optix::make_float3(cosf(theta_s), sinf(theta_s), 0);
    no = ni;
}

// Simple util to do plane ray intersection.
_fn bool intersect_plane(const optix::float3 &plane_origin, const optix::float3 &plane_normal, const optix::Ray &ray,
        float &intersection_distance)
{
    float denom = optix::dot(plane_normal, ray.direction);
    if (fabsf(denom) < 1e-12)
    {
        return false;
    } // Parallel: none or all points of the line lie in the plane.
    intersection_distance = optix::dot((plane_origin - ray.origin), plane_normal) / denom;
    return intersection_distance > ray.tmin && intersection_distance < ray.tmax;
}


_fn optix::uint2 store_values_in_buffer(const optix::uint2& idxs, const float flux_E, BufPtr2D<float> &resulting_flux)
{
    optix_assert(flux_E >= 0.0f);
    optix_assert(!isnan(flux_E));
    optix_print("Storing flux %f\n", flux_E);
    // Atomic add to avoid thread conflicts
    if (!isnan(flux_E))
    {
        atomicAdd(&resulting_flux[idxs], flux_E);
    }
    return idxs;
}


// Returns true if the photon has been absorbed, false otherwise
_fn bool scatter_photon_hemisphere_mcml(OutputShape::Type shape, const BSSRDFSimulatedOptions &options, const BSSRDFRendererData &geometry_data,
        optix::float3 &xp, optix::float3 &wp, float &flux_t, BufPtr2D<float> &resulting_flux,
        const float n2_over_n1, const float albedo, const float extinction, const float g,
        SEED_TYPE &t, int starting_it, int executions)
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
        {
            return true;
        }
#endif
        const float rand = RND_FUNC(t);

        // Sampling new distance for photon, testing if it intersects the interface
        const float d = -logf(rand) / extinction;
        optix::Ray ray = optix::make_Ray(xp, wp, 0, 1e-12f, d);

        if (!isfinite(xp.z))
        {
            return true;
        }

                optix_assert(xp.z <= 1e-6);
                optix_assert(xp.z > -INFINITY);

        optix_print("%d (launch %d) -%f - xp %f %f %f wp %f %f %f \n", i, launch_index.x, optix::dot(wp, no), xp.x,
                xp.y, xp.z, wp.x, wp.y, wp.z);

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
            optix::float3 wo = optix::make_float3(0.0f);
            refract(-wp, -no, n2_over_n1, wo, F_r, cos_theta_i, cos_theta_t);

            const float reflection_probability = RND_FUNC(t);

            if (reflection_probability < F_r)
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
                float theta_s = atan2f(diff.y, diff.x);
                        optix_assert(r >= 0);

                bool is_r_in_bin = r >= geometry_data.mRadius.x && r < geometry_data.mRadius.y;
                bool is_theta_s_in_bin = theta_s >= geometry_data.mThetas.x && theta_s < geometry_data.mThetas.y;
                bool is_p_in_bin = is_r_in_bin && is_theta_s_in_bin;

                float phi_o = atan2f(wo.y, wo.x);
                float theta_o = acosf(wo.z);

                const optix::float2 bins = optix::make_float2(resulting_flux.size());
                optix::float2 coords = get_normalized_hemisphere_buffer_coordinates(shape, phi_o, theta_o);
                optix::uint2 idxs = make_uint2(coords * bins);

                float flux_to_store = flux_t * 1.0f / (geometry_data.mArea);
                float solid_angle_bin = geometry_data.mWeightedSolidAngleBuffer[idxs];

                if(solid_angle_bin > 0.0f)
                    flux_to_store /= solid_angle_bin;
                if(!isfinite(solid_angle_bin) || !isfinite(flux_to_store))
                    printf("ANGLE %f flux %f\n", solid_angle_bin, flux_to_store);

                const float b = options.mBias;
                if (options.mbBiasMode == BiasMode::BIAS_ONLY)
                {
                    const float xoxp = optix::length(surface_point - xp);
                    flux_to_store *= fmaxf(0.0f, 1.0f - b * xoxp * xoxp);
                }
                else if (options.mbBiasMode == BiasMode::BIASED_RESULT)
                {
                    const float xoxp = optix::length(surface_point - xp);
                    const float G = 1.0f / (xoxp * xoxp);
                    flux_to_store *= fminf(G, b) / G;
                }

                if (is_p_in_bin && i > 1) // No single scattering.
                {
                    optix_print("(%d) Refraction. theta_o %f phi_o %f - %f\n", i, theta_o, phi_o, flux_to_store);
                    store_values_in_buffer(idxs, flux_to_store, resulting_flux);
                }
                // We are done with this random walk.
                return true;
            }
        }
    }
    return false;
}

// Returns true if the photon has been absorbed, false otherwise
_fn bool scatter_photon_hemisphere_connections_correct(OutputShape::Type shape, const BSSRDFSimulatedOptions &options,
        const BSSRDFRendererData &geometry_data,
        optix::float3 &xp, optix::float3 &wp,
        float &flux_t,
        BufPtr2D<float> &resulting_flux,
        const float n2_over_n1,
        const float albedo,
        const float extinction, const float g,
        SEED_TYPE &t, int starting_it,
        int executions)
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
        {
            return true;
        }
#endif
        const float rand = RND_FUNC(
                t); // RND_FUNC(t) in [0, 1). Avoids infinity when sampling exponential distribution.

        // Sampling new distance for photon, testing if it intersects the interface
        const float d = -logf(rand) / extinction;
        optix::Ray ray = optix::make_Ray(xp, wp, 0, 1e-12, d);

        if (!isfinite(xp.z))
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
            xp.z = fminf(xp.z,
                    0);  // This is needed to eliminate edge cases that give xp.z > 0 due to numerical precision.

            float r = geometry_data.mRadius.x + geometry_data.mDeltaR * RND_FUNC(t);
            float theta_s = geometry_data.mThetas.x + geometry_data.mDeltaThetas * RND_FUNC(t);

            const optix::float3 xo = xi + r * optix::make_float3(cosf(theta_s), sinf(theta_s), 0);
            optix::float3 w21 = xo - xp;
            const float xoxp = optix::length(w21);
            w21 = w21 / xoxp; // Normalizing

                    optix_assert(optix::dot(w21, no) >= 0.0f);

            // Note: we are checking the *exiting* ray, so we flip the relative ior
            float3 wo = optix::make_float3(0.0f);
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

                float G_prime = 1.0f / (xoxp * xoxp);

                const float b = options.mBias;
                if (options.mbBiasMode == BiasMode::BIAS_ONLY)
                {
                    G_prime = fmaxf(G_prime - b, 0.0f);
                }
                else if (options.mbBiasMode == BiasMode::BIASED_RESULT)
                {
                    G_prime = fminf(G_prime, b);
                }

                const float theta_o = acosf(cos_theta_o);
                optix::float2 bins = optix::make_float2(resulting_flux.size());
                optix::float2 coords = get_normalized_hemisphere_buffer_coordinates(shape, phi_o, theta_o);
                optix::uint2 idxs = make_uint2(coords * bins);

                float geometry_term = fabsf(optix::dot(wo, no)) * G_prime;
                float bssrdf_E = albedo * flux_t * phase_HG(optix::dot(wp, w21), g) * T21 * geometry_term *
                                 expf(-extinction * xoxp);
                bssrdf_E *= r * (2.0f / (geometry_data.mRadius.x + geometry_data.mRadius.y));

                float solid_angle_bin = geometry_data.mWeightedSolidAngleBuffer[idxs];
                if(solid_angle_bin > 0.0f)
                    bssrdf_E /= solid_angle_bin;

                optix_print("flux_t %f, albedo %f, p %f, exp %f, F %f\n", flux_t, albedo,
                        phase_HG(optix::dot(w21, wp), g), expf(-extinction * xoxp), T21);

                // Not including single scattering, so i == 0 is not considered.
                if (i > 0)
                {
                    store_values_in_buffer(idxs, bssrdf_E, resulting_flux);
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

            float3 wo = optix::make_float3(0.0f);
            float F_r = 1.0f;
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

// Old code about bias reduction.
#if 0
if(i > 0 && xoxp < mfp_bias)
{
    optix_print("[bias] zone.\n");
    optix::float3 w12;
    float R12;
    refract(wo, no, 1.0f / n2_over_n1, w12, R12);
    float T12 = (1.0f - R12) ;

    optix_assert(optix::length(w12 + w21) < 1e-3f);

    optix::float3 xs = xo;
    xs.z = 0;
    optix::float3 ws = w12;

    float scattering = albedo * extinction;
    float new_path_flux = 1.0f * T12;

    float G_bias = max(0.0f, 1 - b * xoxp * xoxp);
    float Lo_first = new_path_flux * scattering * phase_HG(dot(-wp, w12),g) * fabsf(dot(w12, no)) * G_bias * expf(-extinction * xoxp);
    Lo_final += Lo_first;
    optix_print("[bias] Accumulating First. %f --> f %f exp %f G_bias %f (original: %f xoxp %f b %f mfp %f) \n", Lo_first, new_path_flux, expf(-extinction * xoxp), G_bias, G, xoxp, b , mfp_bias);

    int safe_count = 0;
    while(true)
    {
        float rand_bias = RND_FUNC(t); // RND_FUNC(t) in [0, 1). Avoids infinity when sampling exponential distribution.
        float d_bias = -log(rand_bias) / extinction;

        optix::float2 smpl = optix::make_float2(RND_FUNC(t), RND_FUNC(t));
        ws = sample_HG(ws, g, smpl);

        optix::Ray ray_bias = optix::make_Ray(xs, ws, 0, 1e-12, d_bias);
        float intersection_distance_bias;

        if(!intersect_plane(xi, ni, ray_bias, intersection_distance_bias))
        {
            xs = xs + ws * d_bias;
            xs.z = fminf(xs.z, 0.0f);

            optix::float3 connection = xp - xs;
            const float xsxp = optix::length(connection);
            connection = normalize(connection);

            if(xsxp > mfp_bias) {
                optix_print("[bias] Too far from xp.\n");
                break;
            }

            G_bias *= max(0.0f, 1.0f - b * xsxp * xsxp);

            const float cosine_term = xs.z == 0.0f? fabsf(dot(wp, no)) : 1.0f;
            //const float cosine_term = 1.0f;

            const float Lo_path = new_path_flux * scattering * phase_HG(dot(-wp, connection),g) * cosine_term * G_bias * expf(-extinction * xsxp);
            Lo_final += Lo_path;
            optix_print("[bias] Accumulating. %f --> f  %f  exp %f G_bias %f (original: %f xsxp %f) \n", Lo_path, new_path_flux, expf(-extinction * xsxp), G_bias, G, xsxp);

            if(Lo_path < Lo * 1e-12f) {
                optix_print("[bias] Too small.\n");
                break;
            }
        }
        else
        {
            const optix::float3 surface_point = xs + ws * intersection_distance_bias;
            float3 refr;
            float F_r;
            refract(-ws, -no, n2_over_n1, refr, F_r);

            // Reflect and turn to face inside.
            new_path_flux *= F_r;
            xs = surface_point;
            ws = reflect(ws, -no);
            xs.z = 0;

            optix_print("[bias] Ray going outside.\n");
        }
        safe_count++;
        if(safe_count >= 10000)
        {
            optix_print("[bias] Quota exceeded.\n");
            break;
        }
    }
}
#endif


// Returns true if the photon has been absorbed, false otherwise
_fn bool scatter_photon_hemisphere_connections(OutputShape::Type shape, const BSSRDFRendererData &geometry_data, optix::float3 &xp, optix::float3 &wp,
        float &flux_t, BufPtr2D<float> &resulting_flux, const float3 &xo,
        const float n2_over_n1, const float albedo, const float extinction, const float g,
        SEED_TYPE &t, int starting_it, int executions)
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
        {
            return true;
        }
#endif
        const float rand =
                1.0f - RND_FUNC(t); // RND_FUNC(t) in [0, 1). Avoids infinity when sampling exponential distribution.

        // Sampling new distance for photon, testing if it intersects the interface
        const float d = -logf(rand) / extinction;
        optix::Ray ray = optix::make_Ray(xp, wp, 0, 1e-12, d);

        if (!isfinite(xp.z))
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
            float3 wo = optix::make_float3(0.0f);
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

                float flux_E =
                        flux_t * scattering * phase_HG(optix::dot(d_vec, wp), g) * expf(-extinction * d_vec_len) * F_t;
#ifdef INCLUDE_GEOMETRIC_TERM
                optix_print("Geometric.\n");
                optix_assert(fabsf(optix::dot(d_vec, no) - cos_theta_21) < 1e-4f);
                flux_E *= fabsf(cos_theta_21) / (d_vec_len*d_vec_len);
#endif
                optix_print("flux_t %f, albedo %f, p %f, exp %f, F %f\n", flux_t, albedo,
                        phase_HG(optix::dot(d_vec, wp), g), expf(-extinction * d_vec_len), F_t);

                // Not including single scattering, so i == 0 is not considered.

#ifdef INCLUDE_SINGLE_SCATTERING
                store_values_in_buffer(cos_theta_o, phi_o, flux_E, resulting_flux);
#else
                if (i > 0)
                {
                    const float theta_o = acosf(cos_theta_o);
                    const optix::size_t2 bins = resulting_flux.size();
                    optix::float2 coords = get_normalized_hemisphere_buffer_coordinates(shape, phi_o, theta_o);
                    optix::uint2 idxs = make_uint2(coords * make_float2(bins));
                    store_values_in_buffer(idxs, flux_E, resulting_flux);
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

            float3 wo = optix::make_float3(0.0f);
            float F_r;
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

_fn optix::float2 get_normalized_planar_buffer_coordinates(const optix::float2 &coord)
{
    return (coord + plane_size / 2) / plane_size;
}

_fn optix::float2 get_planar_buffer_coordinates(const optix::float2 &normalized_coord)
{
    return (normalized_coord * plane_size) - plane_size / 2;
}

// Returns true if the photon has been absorbed, false otherwise
_fn bool scatter_photon(OutputShape::Type shape, const BSSRDFSimulatedOptions &options, const BSSRDFRendererData &geometry_data,
        optix::float3 &xp, optix::float3 &wp, float &flux_t, BufPtr2D<float> &resulting_flux,
        const float3 &xo, const float n2_over_n1, const float albedo, const float extinction,
        const float g, SEED_TYPE &t, int starting_it, int executions)
{
    if (options.mIntegrationMethod == IntegrationMethod::MCML)
    {
        return scatter_photon_hemisphere_mcml(shape, options, geometry_data, xp, wp, flux_t, resulting_flux, n2_over_n1,
                albedo, extinction, g, t, starting_it, executions);
    }
    else if (options.mIntegrationMethod == IntegrationMethod::CONNECTIONS)
    {
        return scatter_photon_hemisphere_connections(shape, geometry_data, xp, wp, flux_t, resulting_flux, xo, n2_over_n1,
                albedo, extinction, g, t, starting_it, executions);
    }
    else if (options.mIntegrationMethod == IntegrationMethod::CONNECTIONS_WITH_FIX)
    {
        return scatter_photon_hemisphere_connections_correct(shape, options, geometry_data, xp, wp, flux_t, resulting_flux,
                n2_over_n1, albedo, extinction, g, t, starting_it,
                executions);
    }
    else
    {
        return false;
    }
}
