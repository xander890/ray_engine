//
// Created by alcor on 1/15/18.
//

#ifndef RAY_ENGINE_SUBSURFACE_SCATTERING_SAMPLED_COMMON_H
#define RAY_ENGINE_SUBSURFACE_SCATTERING_SAMPLED_COMMON_H

#include <device_common.h>
#include <math_utils.h>
#include <random_device.h>
#include <optics_utils.h>
#include <structs.h>
#include <ray_tracing_utils.h>
#include <scattering_properties.h>
#include <material_device.h>
#include <light_device.h>
#include <sampling_helpers.h>
#include <camera_common.h>

#ifndef ENABLE_NEURAL_NETWORK
#include "bssrdf_device.h"
#else
#include <neural_network_sampler_device.h>
#endif

rtDeclareVariable(CameraData, camera_data, , );

_fn bool trace_depth_ray(const optix::float3& origin, const optix::float3& direction, optix::float3 & xi, optix::float3 & normal, const float t_min = scene_epsilon, const float t_max = RT_DEFAULT_MAX)
{
    PerRayData_normal_depth attribute_fetch_ray_payload = { optix::make_float3(0.0f), RT_DEFAULT_MAX };
    optix::Ray attribute_fetch_ray;
    attribute_fetch_ray.ray_type =  RayType::ATTRIBUTE;
    attribute_fetch_ray.tmin = t_min;
    attribute_fetch_ray_payload.depth = t_max;
    attribute_fetch_ray.tmax = t_max;
    attribute_fetch_ray.direction = direction;
    attribute_fetch_ray.origin = origin;

    rtTrace(current_geometry_node, attribute_fetch_ray, attribute_fetch_ray_payload);

    optix_print("Miss? %s, dir %f %f %f\n", abs(attribute_fetch_ray_payload.depth - t_max) < 1e-3 ? "true" : "false", direction.x, direction.y, direction.z);

    if (abs(attribute_fetch_ray_payload.depth - t_max) < 1e-9f) // Miss
        return false;
    xi = origin + attribute_fetch_ray_payload.depth * direction;
    normal = attribute_fetch_ray_payload.normal;
    return true;
}

_fn void sample_r_phi_plane(TEASampler * sampler, const BSSRDFSamplingProperties & bssrdf_sampling_properties, float & r, float & phi, float & pdf)
{
    optix::float2 sample = optix::make_float2(sampler->next1D(), sampler->next1D());
    if(bssrdf_sampling_properties.sampling_tangent_plane_technique == BssrdfSamplePointOnTangentTechnique::EXPONENTIAL_DISK) {
        sample_disk_exponential(sample, bssrdf_sampling_properties.sampling_inverse_mean_free_path, pdf, r, phi);
    }
    else if (bssrdf_sampling_properties.sampling_tangent_plane_technique == BssrdfSamplePointOnTangentTechnique::UNIFORM_DISK)
    {
        sample_disk_uniform(sample, pdf, r, phi, 0.0f, bssrdf_sampling_properties.R_max);
    }
}


_fn void sample_point_on_normal_tangent_plane(
        const float3 & xo,          // The points hit by the camera ray.
        const float3 & no,          // The normal at the point.
        const float3 & wo,          // The incoming ray direction.
        const MaterialDataCommon & material,  // Material properties.
        const BSSRDFSamplingProperties & bssrdf_sampling_properties,
        TEASampler * sampler,       // A rng.
        float3 & x_tangent,                // The candidate point
        float3 & integration_factor, // An factor that will be multiplied into the final result. For inverse pdfs.
        bool & has_candidate_wi,    // Returns true if the point has a candidate outgoing direction
        float3 & proposed_wi)       // The candidate proposed direction.
{
#ifdef ENABLE_NEURAL_NETWORK
        optix_assert(bssrdf_sampling_properties.sampling_tangent_plane_technique == BssrdfSamplePointOnTangentTechnique::NEURAL_NETWORK_IMPORTANCE_SAMPLING);
        has_candidate_wi = true;
        // Sampling neural network with specific colorband.
        // TODO implement Hero wavelength sampling here.
        int colorband = int(sampler->next1D() * 3.0);
        float nn_integration_factor = 1.0f;

        // FIXME remove this when multiple nns are implemented...
        sample_neural_network(xo,no,wo, material, 0, sampler, x_tangent, nn_integration_factor, proposed_wi);
        integration_factor *= nn_integration_factor; // Pdf of choosing wavelength.

        // ...and uncomment this.
        //sample_neural_network(xo,no,wo, material, colorband, sampler, x_tangent, nn_integration_factor, proposed_wi);
        //get_channel(colorband, integration_factor) *= 3.0f * nn_integration_factor; // Pdf of choosing wavelength.
#else
        optix::float3 to, bo;
        create_onb(no, to, bo);
        float r, phi, pdf_disk;
        sample_r_phi_plane(sampler, bssrdf_sampling_properties, r, phi, pdf_disk);
        integration_factor *= make_float3(r / pdf_disk);
        x_tangent = xo + r * cosf(phi) * to + r * sinf(phi) * bo;
        has_candidate_wi = false;
#endif
}


_fn bool sample_xi_ni_from_tangent_hemisphere(const float3 & disc_point, const float3 & disc_normal, float3 & xi, float3 & ni, const float normal_bias = 0.0f, const float t_min = scene_epsilon)
{
    float3 sample_ray_origin = disc_point;
    float3 sample_ray_dir = disc_normal;
    sample_ray_origin += normal_bias * disc_normal; // Offsetting the ray origin along the normal. to shoot rays "backwards" towards the surface
    float t_max = RT_DEFAULT_MAX;
    if (!trace_depth_ray(sample_ray_origin, sample_ray_dir, xi, ni, t_min, t_max))
        return false;

    return true;
}

#ifndef ENABLE_NEURAL_NETWORK
_fn bool camera_based_sampling(const float3 & xo, const float3 & no, const float3 & wo, const MaterialDataCommon & material, const BSSRDFSamplingProperties & bssrdf_sampling_properties, TEASampler * sampler,
                                                      float3 & xi, float3 & ni, float3 & integration_factor)
{
    float r, phi, pdf_disk;
    sample_r_phi_plane(sampler, bssrdf_sampling_properties, r, phi, pdf_disk);

    optix::float3 to, bo;
    create_onb(no, to, bo);
    float t_max = RT_DEFAULT_MAX;
    integration_factor = make_float3(1.0f);
    float3 sample_on_tangent_plane = xo + r * cosf(phi) * to + r * sinf(phi) * bo;
    float3 sample_ray_dir = normalize(sample_on_tangent_plane - camera_data.eye);
    float3 sample_ray_origin = camera_data.eye;

    if (!trace_depth_ray(sample_ray_origin, sample_ray_dir, xi, ni, scene_epsilon, t_max))
        return false;

    integration_factor *= r / pdf_disk;
    optix_print("r: %f, pdf_disk %f, inte %f\n", r, pdf_disk, integration_factor.x);

    if (bssrdf_sampling_properties.use_jacobian == 1)
    {
        float3 d = camera_data.eye - xi;
        float cos_alpha = dot(-sample_ray_dir, ni);

        float3 d_tan = camera_data.eye - sample_on_tangent_plane;
        float cos_alpha_tan = dot(-sample_ray_dir, no);

        float jacobian = max(1e-3, cos_alpha_tan) / max(1e-3, cos_alpha) * max(1e-3, dot(d, d)) / max(1e-3, dot(d_tan, d_tan));
        integration_factor *= jacobian;
    }

    return true;
}
#endif

_fn bool tangent_based_sampling(const float3 & xo, const float3 & no, const float3 & wo, const MaterialDataCommon & material, const BSSRDFSamplingProperties & bssrdf_sampling_properties, TEASampler * sampler,
                                                       float3 & xi, float3 & ni, float3 & integration_factor, bool & has_candidate_wi, float3 & proposed_wi)
{

    optix::float3 xo_tangent;
    sample_point_on_normal_tangent_plane(xo,no,wo,material, bssrdf_sampling_properties, sampler, xo_tangent, integration_factor, has_candidate_wi, proposed_wi);

    if (!sample_xi_ni_from_tangent_hemisphere(xo_tangent, -no, xi, ni, -bssrdf_sampling_properties.d_max))
        return false;

    float inv_jac = max(bssrdf_sampling_properties.dot_no_ni_min, dot(normalize(no), normalize(ni)));

    if (bssrdf_sampling_properties.use_jacobian == 1)
        integration_factor *= inv_jac > 0.0f ? 1 / inv_jac : 0.0f;
    return true;
}

#ifndef ENABLE_NEURAL_NETWORK
_fn bool axis_mis_probes(const float3 & xo, const float3 & no, const float3 & wo, const MaterialDataCommon & material, const BSSRDFSamplingProperties & bssrdf_sampling_properties, TEASampler * sampler, float3 & xi, float3 & ni, float3 & integration_factor)
{
    float chosen_sampling_mfp =  bssrdf_sampling_properties.sampling_inverse_mean_free_path;
    float r, phi, pdf_disk;
    sample_r_phi_plane(sampler, bssrdf_sampling_properties, r, phi, pdf_disk);

    optix::float3 to, bo;
    create_onb(no, to, bo);

    optix::float3 axes[3] = { no, bo, to };

    int main_axis = sampler->next1D() * 3.0f;
    float inv_pdf_axis = 3.0f;

    float verse = sampler->next1D() < 0.5f? -1 : 1;

    float3 probe_direction = verse * axes[main_axis];
    float inv_pdf = 2.0f * inv_pdf_axis;

    optix::float3 chosen_axes[3] = { probe_direction, make_float3(0), make_float3(0) };
    create_onb(probe_direction, chosen_axes[1], chosen_axes[2]);

    float3 xi_tangent_space = xo + chosen_axes[1]* r * cosf(phi)  + chosen_axes[2]* r * sinf(phi);
    float3 sample_ray_dir = probe_direction;
    float t_max = RT_DEFAULT_MAX;

    if (!trace_depth_ray(xi_tangent_space + no * scene_epsilon * 2, sample_ray_dir, xi, ni, 0.0f, t_max))
        return false;

    optix::float3 xo_xi = xo - xi;

    float dot0 = abs(dot(ni, probe_direction));

    float3 axis_1 = chosen_axes[1];
    float dot1 = abs(dot(ni, axis_1));
    float3 axis_2 = chosen_axes[2];
    float dot2 = abs(dot(ni, axis_2));

    float wi0 = pdf_disk * dot0 / r;

    float3 tangent_point_xi_xo_1 = xo_xi - dot(xo_xi, axis_1) * axis_1;
    float r_axis_1 = optix::length(tangent_point_xi_xo_1);
    float pdf_axis_1 = exponential_pdf_disk(r_axis_1, chosen_sampling_mfp);
    float wi1 = pdf_axis_1 * dot1 / r_axis_1;

    float3 tangent_point_xi_xo_2 = xo_xi - dot(xo_xi, axis_2) * axis_2;
    float r_axis_2 = optix::length(tangent_point_xi_xo_2);
    float pdf_axis_2 = exponential_pdf_disk(r_axis_2, chosen_sampling_mfp);
    float wi2 = pdf_axis_2 * dot2 / r_axis_2;

    float weight = 1.0f / (wi0 + wi1 + wi2);

    integration_factor = make_float3(inv_pdf * weight);
    return true;
}
#endif

_fn bool importance_sample_position(const float3 & xo, const float3 & no, const float3 & wo, const MaterialDataCommon & material, const BSSRDFSamplingProperties & bssrdf_sampling_properties, TEASampler * sampler,
                                                           float3 & xi, float3 & ni, float3 & integration_factor, bool & has_candidate_wi, float3 & proposed_wi)
{
    has_candidate_wi = false;
    switch (bssrdf_sampling_properties.sampling_method)
    {
        case BssrdfSamplingType::BSSRDF_SAMPLING_TANGENT_PLANE:				return tangent_based_sampling(xo, no, wo, material, bssrdf_sampling_properties, sampler, xi, ni, integration_factor, has_candidate_wi, proposed_wi);
#ifndef ENABLE_NEURAL_NETWORK
        case BssrdfSamplingType::BSSRDF_SAMPLING_CAMERA_BASED:				return camera_based_sampling(xo, no, wo, material, bssrdf_sampling_properties, sampler, xi, ni, integration_factor);
        case BssrdfSamplingType::BSSRDF_SAMPLING_MIS_AXIS:					return axis_mis_probes(xo, no, wo, material, bssrdf_sampling_properties, sampler, xi, ni, integration_factor);
#endif
    }
    return false;
}

#endif //RAY_ENGINE_SUBSURFACE_SCATTERING_SAMPLED_COMMON_H
