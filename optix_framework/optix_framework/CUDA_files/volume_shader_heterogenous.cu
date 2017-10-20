#include <device_common_data.h>
#include <random.h>
#include <sampling_helpers.h>
#include <optical_helper.h>
#include <environment_map.h>
#include "structs.h"
#include "scattering_properties.h"
#include "light.h"
#include <material_device.h>
#include "phase_function.h"

using namespace optix;

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(unsigned int , maximum_volume_steps, , );


__device__ __forceinline__ float get_volume_step()
{   
    float m = local_bounding_box->maxExtent();
    float max_voxels = 2.0f * rtTexSize(noise_tex).x;
    float step = m / max_voxels;
    return step;
}

__device__ __inline__ float get_extinction(const optix::float3 & pos, int colorband)
{
    const ScatteringMaterialProperties& props = get_material(pos).scattering_properties;
    return *(&props.extinction.x + colorband);
}

__device__ __inline__ float get_albedo(const optix::float3 & pos, int colorband)
{
    const ScatteringMaterialProperties& props = get_material(pos).scattering_properties;
    return *(&props.albedo.x + colorband);
}

__device__ __inline__ float get_asymmetry(const optix::float3 & pos, int colorband)
{
    const ScatteringMaterialProperties& props = get_material(pos).scattering_properties;
    return *(&props.meancosine.x + colorband);
}

__device__ __inline__ bool scatter_inside(optix::Ray& ray, int colorband, uint& t)
{
    // Input: 
    // ray: initial position and direction
    //
    // Output:
    // ray: position and direction at the last vertex before intersecting
    // the volume boundary from inside
    //
    // return value: 
    //  true: scattered through the volume
    //  false: absorbed
    //
    ray.tmin = scene_epsilon;
    ray.ray_type =  RayType::DEPTH;
    PerRayData_depth prd_ray;
    prd_ray.depth = 0.0f;
    
    ray.tmax = RT_DEFAULT_MAX;

    float albedo = get_albedo(ray.origin, colorband);
    float g = get_asymmetry(ray.origin, colorband);
    float extinction = get_extinction(ray.origin, colorband);
    float delta_t = get_volume_step();

	for (int k = 0; k < maximum_volume_steps; k++)
    {
        // Sample new distance
        rtTrace(top_object, ray, prd_ray); // Calculating depth of the ray.
        extinction = get_extinction(ray.origin, colorband);
        float s = prd_ray.depth;
        float log_xi = log(rnd(t));

#define NEW_HETEROGENOUS
#ifdef NEW_HETEROGENOUS
        int N_samples = (int)floor(s / delta_t);
        float tau = 0.0f;
        float offset = rnd(t);
        float ti = offset / N_samples;
        
        // Ray marching to find the real optical depth for heterogenous materials.
        int i = 0;
        for (i = 1; i < N_samples; i++)
        {
            ti += i / ((float)N_samples);
            float3 pos = ray.origin + ti * ray.direction;
            extinction = get_extinction(pos, colorband);
            float new_tau = tau + extinction * delta_t; // next optical depth
            if (new_tau + log_xi > 0)
                break;
            tau = new_tau; // We update only here, so if the loop breaks we have tau(t-1, s) stored.
        }
       
        float true_optical_distance = (i - 1)*delta_t - (tau + log_xi) / extinction; // (tau + log_xi) is negative, so subtracting it we get a positive.

        float sampled_distance = true_optical_distance;
#else
        float sampled_distance = -log_xi / extinction;
#endif
        if (sampled_distance < s)
        {
            // we have found a suitable optical distance, i.e., still inside the volume

            // New ray origin
            ray.origin += ray.direction * sampled_distance;

            // New ray direction 
            g = get_asymmetry(ray.origin, colorband);
            ray.direction = sample_HG(ray.direction, g, t);
        }
        else // Intersection hit
            return true;

        albedo = get_albedo(ray.origin, colorband);
        // Break if absorbed
        if (rnd(t) > albedo)
            return false;
    }
}


// Any hit program for checking that we are still inside
RT_PROGRAM void any_hit_shadow()
{
    // this material is opaque, so it fully attenuates all shadow rays	
    //rtPrintf("any hit %f\n", t_hit);
    prd_shadow.attenuation = 0.0f;
    rtTerminateRay();
}


RT_PROGRAM void shade()
{
    prd_radiance.result = make_float3(0.0f);
    if (prd_radiance.depth > max_depth) return;

    // Compute cosine to angle of incidence
    float3 hit_pos = ray.origin + t_hit*ray.direction;
    float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 w_i = -ray.direction;
    const MaterialDataCommon & material = get_material();
    const ScatteringMaterialProperties& props = material.scattering_properties;
    float n1_over_n2 = 1.0f / material.relative_ior;
    float cos_theta_in = dot(normal, w_i);
    float3 beam_T = make_float3(1.0f);
    uint& t = prd_radiance.seed;

    // Russian roulette with absorption if arrived from dense medium
    bool inside = cos_theta_in < 0.0f;
    if (inside)
    {
        n1_over_n2 = material.relative_ior;
        normal = -normal;
        cos_theta_in = -cos_theta_in;
    }
    else if (material.relative_ior < 1.0f)
    {
        beam_T = expf(-t_hit*props.absorption);
        float prob = (beam_T.x + beam_T.y + beam_T.z) / 3.0f;
        if (rnd(t) >= prob) return;
        beam_T /= prob;
    }

    // Compute Fresnel reflectance (R) and trace refracted ray if necessary
	float R;
	float3 reflected_dir = -reflect(w_i, normal);
	float3 refracted_dir;
	refract(w_i, normal, n1_over_n2, refracted_dir, R);

    // Sample new ray inside or outside
    ++prd_radiance.depth;
    prd_radiance.flags |= RayFlags::USE_EMISSION;
    float xi = rnd(t);
    if ((xi < R && inside) || (xi > R && !inside))
    {
        // Sample a color and set properties
        int colorband;
        float weight;
        if (prd_radiance.colorband == -1)
        {
            colorband = int(3.0f*rnd(t));
            prd_radiance.colorband = colorband;
            weight = 3.0;
        }
        else
        {
            colorband = prd_radiance.colorband;
            weight = 1.0f;
        }


        // Trace inside, i.e., reflect from inside or refract from outside
        float3 dir_inside = inside ? reflected_dir : refracted_dir;
        Ray ray_inside(hit_pos, dir_inside,  RayType::SHADOW, scene_epsilon, RT_DEFAULT_MAX);

        if (scatter_inside(ray_inside, colorband, t))
        {
            // Switch to radiance ray and intersect with boundary
            ray_inside.ray_type =  RayType::RADIANCE;
            ray_inside.tmax = RT_DEFAULT_MAX;
            rtTrace(top_object, ray_inside, prd_radiance);

            *(&prd_radiance.result.x + colorband) *= weight*(*(&beam_T.x + colorband));
            *(&prd_radiance.result.x + (colorband + 1) % 3) = 0.0f;
            *(&prd_radiance.result.x + (colorband + 2) % 3) = 0.0f;
        }
    }
    else
    {
        // Trace outside, i.e., refract from inside or reflect from outside
        float3 dir_outside = inside ? refracted_dir : reflected_dir;
        Ray ray_outside(hit_pos, dir_outside,  RayType::RADIANCE, scene_epsilon, RT_DEFAULT_MAX);
        rtTrace(top_object, ray_outside, prd_radiance);
        prd_radiance.result = beam_T * (prd_radiance.result);

    }
}
