// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <color_helpers.h>
#include <environment_map.h>
#include <material_device.h>
#include <ray_trace_helpers.h>
#include <default_shader_common.h>
#include <microfacets.h>

using namespace optix;

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow,   prd_shadow,   rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float2, texcoord, attribute texcoord, );
//rtDeclareVariable(InterfaceVisualizationFlags::Type, options, ,);

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() { 
    const MaterialDataCommon & material = get_material(texcoord);
    float3 emission = make_float3(rtTex2D<float4>(material.ambient_map, texcoord.x, texcoord.y));
 shadow_hit(prd_shadow, emission);
}

_fn void _shade()
{

    const MaterialDataCommon & material = get_material(texcoord);
    float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 wo = -ray.direction;
    float3 hit_pos = ray.origin + t_hit * ray.direction;
    TEASampler& sampler = *prd_radiance.sampler;

    if (prd_radiance.depth < max_depth)
    {

        float3 beam_T = make_float3(1.0f);
        if (dot(wo, normal) < 0.0f)
        {
            beam_T = expf(-t_hit*material.scattering_properties.absorption);
            float prob = (beam_T.x + beam_T.y + beam_T.z) / 3.0f;
            if (sampler.next1D() >= prob)
            {
                prd_radiance.result = make_float3(0);
                return;
            }
            beam_T /= fmaxf(10e-6f,prob);
        }

        PerRayData_radiance prd = prd_radiance;
        prd.depth = prd_radiance.depth + 1;
        prd.flags |= RayFlags::USE_EMISSION;
        float3 n = normal;

        optix::float3 m = importance_sample_ggx(sampler.next2D(), n, material.roughness);
        optix::Ray reflected, refracted;
        float R, cos_theta_signed;
        optix::float3 ff_m;
        float relative_ior = dot(material.index_of_refraction, optix::make_float3(1)) / 3.0f;
        get_glass_rays(wo, relative_ior, make_float3(0), m, ff_m, reflected, refracted, R, cos_theta_signed);

        float xi = sampler.next1D();
        float3 wi = (xi < R)? reflected.direction : refracted.direction;

        optix::Ray ray_t = optix::make_Ray(hit_pos, wi, RayType::RADIANCE, scene_epsilon, RT_DEFAULT_MAX);
        rtTrace(top_object, ray_t, prd);

        optix::float3 ff_normal = faceforward(n, wo, n);
        float G = ggx_G1(wi, ff_m, ff_normal, material.roughness) * ggx_G1(wo, ff_m, ff_normal, material.roughness);
        float weight = dot(wo, ff_m) / (dot(wo, n) * dot(n, ff_m)) * G;
        float importance_sampled_brdf = abs(weight);
        prd_radiance.result = prd.result * importance_sampled_brdf * beam_T;
    }
    else
    {
        prd_radiance.result = make_float3(0.0f);
    }

}

RT_PROGRAM void shade() { _shade(); }
RT_PROGRAM void shade_path_tracing() { _shade(); }