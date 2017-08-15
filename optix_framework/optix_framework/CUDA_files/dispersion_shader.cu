// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <device_common_data.h>
#include <color_helpers.h>
#include <environment_map.h>
#include <material.h>
using namespace optix;

//#define USE_SPECTRAL_RENDERING

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );

rtDeclareVariable(float3, ior_complex_real_sq, , );
rtDeclareVariable(float3, ior_complex_imag_sq, , );

rtDeclareVariable(MaterialDataCommon, material, , );

// Russian roulette variables
rtDeclareVariable(int, max_splits, , );

rtBuffer<float3, 1> normalized_cie_rgb; 
rtBuffer<float, 1> normalized_cie_rgb_cdf;
rtDeclareVariable(float, normalized_cie_rgb_step, , );
rtDeclareVariable(float, normalized_cie_rgb_wavelength, , );

rtBuffer<float, 1> ior_real_spectrum;
rtDeclareVariable(float, ior_real_wavelength, , );
rtDeclareVariable(float, ior_real_step, , );

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow() {
	float3 emission = make_float3(rtTex2D<float4>(material.ambient_map, texcoord.x, texcoord.y));

	shadow_hit(prd_shadow, emission);
}

__forceinline__ __device__ unsigned int cdf_bsearch(float xi)
{
	uint table_size = normalized_cie_rgb_cdf.size();
	uint middle = table_size = table_size >> 1;
	uint odd = 0;
	while (table_size > 0)
	{
		odd = table_size & 1;
		table_size = table_size >> 1;
		unsigned int tmp = table_size + odd;
		middle = xi > normalized_cie_rgb_cdf[middle] ? middle + tmp : (xi < normalized_cie_rgb_cdf[middle - 1] ? middle - tmp : middle);
	}
	return middle;
}

// Closest hit program for Lambertian shading using the basic light as a directional source + specular term (blinn phong)
RT_PROGRAM void shade()
{
	float3 color = make_float3(0.0f);


	if (prd_radiance.depth < max_depth)
	{
		float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
		float3 ffnormal = faceforward(normal, -ray.direction, normal);
		float3 hit_pos = ray.origin + t_hit * ray.direction;


		PerRayData_radiance prd_refract;
		prd_refract.depth = prd_radiance.depth + 1;
		prd_refract.flags = prd_radiance.flags | RayFlags::USE_EMISSION;
		prd_refract.colorband = prd_radiance.colorband;

		PerRayData_radiance prd_refl;
		prd_refl.depth = prd_radiance.depth + 1;
		prd_refl.flags = prd_radiance.flags | RayFlags::USE_EMISSION;

		prd_refl.colorband = prd_radiance.colorband;

		Ray reflected_ray, refracted_ray;
		float R, cos_theta;
		get_glass_rays(ray, material.ior, hit_pos, normal, reflected_ray, refracted_ray, R, cos_theta);

		rtTrace(top_object, reflected_ray, prd_refl);
		color += R * prd_refl.result;
		rtTrace(top_object, refracted_ray, prd_refract);
		color += (1 - R) * prd_refract.result;

	}
	prd_radiance.result = color;
}

__device__ __forceinline__ float& get_band(optix::float3 & v, int band)
{
	return *(&v.x + band);
}

RT_PROGRAM void shade_path_tracing(void)
{

	float3 color = make_float3(0.0f);

	if (prd_radiance.depth < max_depth)
	{
		float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
		float3 hit_pos = ray.origin + t_hit*ray.direction;
		hit_pos = rtTransformPoint(RT_OBJECT_TO_WORLD, hit_pos);
		uint t = prd_radiance.seed;

		PerRayData_radiance prd_new_ray;
		prd_new_ray.depth = prd_radiance.depth + 1;
		prd_new_ray.flags = prd_radiance.flags | RayFlags::USE_EMISSION;

		float3 spectral_color = make_float3(0.0f);
		float w = 0.0f;

#ifdef USE_SPECTRAL_RENDERING
		int band = 0;
		float3 c_m;

		if (prd_radiance.colorband == -1)
		{
			// Sampling a wavelenght using CIE color matching functions CDF
			float rand = rnd(t);
			unsigned int c = cdf_bsearch(rand);
			float lambda = normalized_cie_rgb_wavelength + c * normalized_cie_rgb_step;
			band = clamp( (int)floor((lambda - ior_real_wavelength) / ior_real_step), 0, ior_real_spectrum.size() -1);
			float invpdf = c > 0
				? 1.0f / (normalized_cie_rgb_cdf[c] - normalized_cie_rgb_cdf[c - 1])
				: 1.0f / normalized_cie_rgb_cdf[c];
			c_m = normalized_cie_rgb[c] * invpdf;
		}
		else
		{
			// continue using the same frequency.
			band = prd_radiance.colorband;
			c_m = make_float3(1);
		}

		float index_of_refraction = ior_real_spectrum[band];
#else
		int band = 0;
		
		if (prd_radiance.colorband == -1)
		{
			// Selecting a random color channel.
			band = int(rnd(t) * 3.0f);
			w = 3.0f;
		}
		else
		{
			// Continue using the same colorband.
			band = prd_radiance.colorband;
			w = 1.0f;
		}

		float index_of_refraction = sqrt(get_band(ior_complex_real_sq, band));
#endif
		// Setting up payload and glass rays.
		Ray reflected_ray, refracted_ray;
		float R, cos_theta;
		get_glass_rays(ray, index_of_refraction, hit_pos, normal, reflected_ray, refracted_ray, R, cos_theta);
		prd_new_ray.depth = prd_radiance.depth + 1;
		prd_new_ray.flags = prd_radiance.flags | RayFlags::USE_EMISSION;
		prd_new_ray.colorband = band;

		// Glass russian roulette
		float xi = rnd(t);
		prd_new_ray.seed = t;
		if (xi < R)
		{
			rtTrace(top_object, reflected_ray, prd_new_ray);
			spectral_color = prd_new_ray.result;
		}
		else
		{
			rtTrace(top_object, refracted_ray, prd_new_ray);
			spectral_color = prd_new_ray.result;
		}
		

		// Combine the final result.
#ifdef USE_SPECTRAL_RENDERING
		color += c_m * spectral_color;
#else
		float component = get_band(spectral_color, band);
		get_band(color, band) = component;
#endif
		prd_radiance.seed = prd_new_ray.seed;
	}


	prd_radiance.result = color;
}
