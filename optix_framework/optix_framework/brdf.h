#pragma once
#include <device_common_data.h>
#include "brdf_properties.h"
#include "material.h"
#include "sampler.h"
#include <merl_common.h>
#include "microfacets.h"
#include "ray_trace_helpers.h"

rtDeclareVariable(BRDFType::Type, selected_brdf, , );

rtDeclareVariable(optix::float3, merl_brdf_multiplier, , );
rtDeclareVariable(BufPtr1D<float>, merl_brdf_buffer, ,);

// Inline function to evaluate the BRDF.
_fn optix::float3 brdf(const BRDFGeometry & geometry,
        const MaterialDataCommon& material, TEASampler & sampler)
{
    optix::float3 f = optix::make_float3(0);
    const float relative_ior = dot(material.index_of_refraction, optix::make_float3(1)) / 3.0f;

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
            f = f_d + torrance_sparrow_brdf(geometry.n, normalize(geometry.wi), normalize(geometry.wo), relative_ior, material.roughness) * k_s;
        }
            break;
        case BRDFType::GGX:
        {
            optix::float3 k_d = make_float3(optix::rtTex2D<optix::float4>(material.diffuse_map, geometry.texcoord.x, geometry.texcoord.y));
            optix::float3 f_d = k_d * M_1_PIf;
            f = f_d * walter_brdf(geometry.n, normalize(geometry.wi), normalize(geometry.wo), NormalDistribution::GGX, relative_ior, material.roughness);
        }
            break;
        case BRDFType::MERL:
        {
            auto merl = lookup_brdf_val(merl_brdf_buffer, geometry.n, geometry.wi, geometry.wo);
            optix_print("Mult %f %f %f, val %f %f %f\n", merl_brdf_multiplier.x, merl_brdf_multiplier.y, merl_brdf_multiplier.z, merl.x, merl.y, merl.z);
            f = merl_brdf_multiplier * merl;
        }

        break;
        case BRDFType::NotValidEnumItem:
            break;
    }
    return f;
}

// Importance samples a new direction for the brdf, providing appropriate pdf-weighted brdf.
_fn void importance_sample_new_direction_brdf(BRDFGeometry & geometry,
        const MaterialDataCommon& material, TEASampler & sampler, optix::float3 & new_direction, optix::float3 & importance_sampled_brdf)
{
    switch (selected_brdf)
    {
        case BRDFType::LAMBERTIAN:
        case BRDFType::TORRANCE_SPARROW:
        case BRDFType::MERL:
        {
			// We can't do much better in these cases that cosine sample an hemisphere.
            geometry.wi = sample_hemisphere_cosine(sampler.next2D(), geometry.n);
            new_direction = geometry.wi;
            importance_sampled_brdf = brdf(geometry, material, sampler) * M_PIf;
        }
            break;
        case BRDFType::GGX:
        {
            optix::float3 k_d = make_float3(optix::rtTex2D<optix::float4>(material.diffuse_map, geometry.texcoord.x, geometry.texcoord.y));
            importance_sampled_brdf = k_d;

            optix::float3 m = importance_sample_ggx(sampler.next2D(), geometry.n, material.roughness);
            optix::Ray reflected, refracted;
            float R, cos_theta_signed;
            optix::float3 ff_m;
            const float relative_ior = dot(material.index_of_refraction, optix::make_float3(1)) / 3.0f;
            get_glass_rays(geometry.wo, relative_ior, make_float3(0), m, ff_m, reflected, refracted, R, cos_theta_signed);

            new_direction = reflected.direction;
            geometry.wi = new_direction;

            optix::float3 ff_normal = faceforward(geometry.n, geometry.wo, geometry.n);
            float G = ggx_G1(geometry.wi, ff_m, ff_normal, material.roughness) * ggx_G1(geometry.wo, ff_m, ff_normal, material.roughness);
            float weight = dot(geometry.wo, ff_m) / (dot(geometry.wo, geometry.n) * dot(geometry.n, ff_m)) * G;// * R;
            importance_sampled_brdf *= abs(weight);
        }
            break;
        case BRDFType::NotValidEnumItem:
            break;
    }
}