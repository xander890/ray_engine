#pragma once
#include "brdf_common.h"
#include "material_common.h"
#include "sampler_device.h"
#include "merl_common.h"
#include "microfacet_utils.h"
#include "ray_tracing_utils.h"
#include "device_common.h"
#include "brdf_ridged_qr_device.h"

// Type of brdf currently selected.
rtDeclareVariable(BRDFType::Type, selected_brdf, , );

// Common MERL data.
rtDeclareVariable(optix::float3, merl_brdf_multiplier, , );
rtDeclareVariable(BufPtr1D<float>, merl_brdf_buffer, ,);

//
// Evaluates the BRDF. Pass local geometry, a material and a RNG.
//
_fn optix::float3 brdf(const BRDFGeometry & geometry, const MaterialDataCommon& material, TEASampler & sampler)
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
		case BRDFType::BECKMANN:
		{
            optix::float3 k_d = make_float3(optix::rtTex2D<optix::float4>(material.diffuse_map, geometry.texcoord.x, geometry.texcoord.y));
            importance_sampled_brdf = k_d;

			optix::float3 m;
			if(selected_brdf == BRDFType::BECKMANN)
			{
				m = importance_sample_beckmann(sampler.next2D(), geometry.n, material.roughness);
			}
			else
			{
				m = importance_sample_ggx(sampler.next2D(), geometry.n, material.roughness);
			}

            optix::Ray reflected, refracted;
            float R, cos_theta_signed;
            optix::float3 ff_m;
            const float relative_ior = dot(material.index_of_refraction, optix::make_float3(1)) / 3.0f;
            get_glass_rays(geometry.wo, relative_ior, optix::make_float3(0), m, ff_m, reflected, refracted, R, cos_theta_signed);

            new_direction = reflected.direction;
            geometry.wi = new_direction;

            optix::float3 ff_normal = optix::faceforward(geometry.n, geometry.wo, geometry.n);

      		float G;
			if (selected_brdf == BRDFType::BECKMANN)
			{
				G = beckmann_G1_approx(geometry.wi, ff_m, ff_normal, material.roughness) * beckmann_G1_approx(geometry.wo, ff_m, ff_normal, material.roughness);
			}
			else
			{
				G = ggx_G1(geometry.wi, ff_m, ff_normal, material.roughness) * ggx_G1(geometry.wo, ff_m, ff_normal, material.roughness);
			}
			
            float weight = dot(geometry.wo, ff_m) / (dot(geometry.wo, geometry.n) * dot(geometry.n, ff_m)) * G;// * R;
            importance_sampled_brdf *= abs(weight);
        }
            break;
		case BRDFType::QR_RIDGED:
		{
			importance_sample_qr_brdf(geometry, material, sampler, new_direction, importance_sampled_brdf);
		}
        case BRDFType::NotValidEnumItem:
            break;
    }
}