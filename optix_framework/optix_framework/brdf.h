#pragma once
#include <device_common_data.h>
#include "brdf_properties.h"
#include "material.h"
#include "sampler.h"

rtDeclareVariable(BRDFType::Type, selected_brdf, , );

#include <merl_common.h>

rtDeclareVariable(optix::float3, merl_brdf_multiplier, , );
rtDeclareVariable(BufPtr1D<float>, merl_brdf_buffer,,);

__device__ __inline__
float geometric_term_torrance_sparrow(const optix::float3& n, const optix::float3& wi, const optix::float3& wo, const optix::float3& wh)
{
    float n_dot_h = fabsf(dot(n, wh));
    float n_dot_o = fabsf(dot(n, wo));
    float n_dot_i = fabsf(dot(n, wi));
    float i_dot_o = fabsf(dot(wo, wh));
    float min_io = fminf(n_dot_o, n_dot_i);
    return fminf(1.0f, min_io * n_dot_h * 2.0f / i_dot_o);
}


__device__ __inline__
float blinn_microfacet_distribution(const float shininess, const optix::float3& n, const optix::float3& brdf_normal)
{
    float cos_theta = fabsf(dot(n, brdf_normal));
    float D = 0.5f * M_1_PIf * (shininess + 2.0f) * powf(cos_theta, shininess);
    return D;
}

__device__ __inline__ float torrance_sparrow_brdf(const optix::float3 & n, const optix::float3 & wi, const optix::float3 & wo, float ior, float shininess)
{
    float cos_o = dot(n, wo);
    float cos_i = dot(n, wi);
    optix::float3 brdf_normal = normalize(wi + wo);
    float cos_brdf_normali = dot(wi, brdf_normal);
    float cos_brdf_normalo = dot(wo, brdf_normal);
    if (cos_brdf_normalo / cos_o <= 0.0f || cos_brdf_normali / cos_i <= 0.0f)
        return 0.0f;

    float D = blinn_microfacet_distribution(shininess, n, brdf_normal);
    float G = geometric_term_torrance_sparrow(n, wi, wo, brdf_normal);
    float F = fresnel_R(cos_o, ior);
    float S = 4.0f * cos_o * cos_i;
    return fabsf(D * F * G / S);
}

__forceinline__ __device__ optix::float3 brdf(const BRDFGeometry & geometry, const float recip_ior,
        const MaterialDataCommon& material, TEASampler & sampler)
{
    optix::float3 f = optix::make_float3(0);
    switch (selected_brdf)
    {
        case BRDFType::LAMBERTIAN:
        {
            optix::float3 k_d = make_float3(optix::rtTex2D<optix::float4>(material.diffuse_map, geometry.texcoord.x, geometry.texcoord.y));
            f = k_d * M_1_PIf;
        }
            break;
        case BRDFType::TORRANCE_SPARROW:
        {
            optix::float3 k_d = make_float3(optix::rtTex2D<optix::float4>(material.diffuse_map, geometry.texcoord.x, geometry.texcoord.y));
            optix::float3 k_s = make_float3(optix::rtTex2D<optix::float4>(material.specular_map, geometry.texcoord.x, geometry.texcoord.y));
            optix::float3 f_d = k_d * M_1_PIf;
            f = f_d +
                torrance_sparrow_brdf(geometry.n, normalize(geometry.wi), normalize(geometry.wo), recip_ior, material.shininess) *
                k_s;
        }
            break;
        case BRDFType::MERL:
        {
            f = merl_brdf_multiplier * lookup_brdf_val(merl_brdf_buffer, geometry.n, geometry.wi, geometry.wo);
        }

        break;
        case BRDFType::NotValidEnumItem:
            break;
    }
    return f;
}