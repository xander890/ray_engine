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
#include "volume_path_tracing_common.h"
#include "bssrdf.h"

using namespace optix;

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

// Material properties
 
rtDeclareVariable(unsigned int, N, , );
rtDeclareVariable(unsigned int, maximum_volume_steps, , );
rtDeclareVariable(unsigned int, volume_pt_mode, , );

//#define USE_SIMILARITY
#define SIMILARITY_STEPS 10

__device__ __inline__ float get_extinction(const optix::float3 & pos, int colorband)
{
    const ScatteringMaterialProperties& props = get_material(pos).scattering_properties;
    return *(&props.extinction.x + colorband);
}

__device__ __inline__ float get_reduced_extinction(const optix::float3 & pos, int colorband)
{
    const ScatteringMaterialProperties& props = get_material(pos).scattering_properties;
    return *(&props.reducedExtinction.x + colorband);
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

__device__ __inline__ bool scatter_inside(optix::Ray& ray, int colorband, TEASampler * sampler, int & scattering_events)
{
    float albedo = get_albedo(ray.origin, colorband);
    float g = get_asymmetry(ray.origin, colorband);
    float extinction = get_extinction(ray.origin, colorband);
    optix_print("albedo %f, g %f, ext %f \n", albedo, g, extinction);
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
    ray.ray_type =  RayType::SHADOW;
    PerRayData_shadow prd_ray;
    prd_ray.attenuation = 1.0f;

    for (scattering_events = 0; scattering_events < maximum_volume_steps; scattering_events++)
    {
        // Sample new distance
        ray.tmax = -logf(sampler->next1D()) / extinction;

        rtTrace(top_shadower, ray, prd_ray);
        if (prd_ray.attenuation > 0.0f)
        {
            // The shadow ray did not hit anything, i.e., still inside the volume

            // New ray origin
            ray.origin += ray.direction * ray.tmax;

            // New ray direction
            ray.direction = sample_HG(ray.direction, g, sampler->next2D());
        }
        else // Intersection hit
        {
            return true;
        }

        // Break if absorbed
        if (sampler->next1D() > albedo)
            return false;
    }
    return false;

}


// Any hit program for checking that we are still inside
RT_PROGRAM void any_hit_shadow()
{
    // this material is opaque, so it fully attenuates all shadow rays	
    //rtPrintf("any hit %f\n", t_hit);
    prd_shadow.attenuation = 0.0f;
    rtTerminateRay();
}


// Closest hit program
RT_PROGRAM void shade()
{
    prd_radiance.result = make_float3(0.0f);
    if (prd_radiance.depth > max_depth) return;

    // Compute cosine to angle of incidence
    float3 hit_pos = ray.origin + t_hit*ray.direction;
    float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 w_i = -ray.direction;
    const MaterialDataCommon & material = get_material();
    float n1_over_n2 = 1.0f / material.relative_ior;
    float cos_theta_in = dot(normal, w_i);

    // Russian roulette with absorption if arrived from dense medium
    bool inside = cos_theta_in < 0.0f;
    if (inside)
    {
        n1_over_n2 = material.relative_ior;
        normal = -normal;
    }

    // Compute Fresnel reflectance (R) and trace refracted ray if necessary
    float R;
	float3 reflected_dir = -reflect(w_i, normal); 
    float3 refracted_dir;
	refract(w_i, normal, n1_over_n2, refracted_dir, R);

    // Sample new ray inside or outside
    ++prd_radiance.depth;
    prd_radiance.flags |= RayFlags::USE_EMISSION;
    float xi = prd_radiance.sampler->next1D();
    if ((xi < R && inside) || (xi > R && !inside))
    {
        // Sample a color and set properties
        int colorband;
        float weight;
        if (prd_radiance.colorband == -1)
        {
            colorband = int(3.0f*prd_radiance.sampler->next1D());
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

        int scattering_events = 0;

        if (scatter_inside(ray_inside, colorband, prd_radiance.sampler, scattering_events))
        {
            if(volume_pt_mode == VOLUME_PT_SINGLE_SCATTERING_ONLY && scattering_events >= 2)
                return;
            if(volume_pt_mode == VOLUME_PT_MULTIPLE_SCATTERING_ONLY && scattering_events < 2)
                return;

            // Switch to radiance ray and intersect with boundary
            ray_inside.ray_type =  RayType::RADIANCE;
            ray_inside.tmax = RT_DEFAULT_MAX;
            rtTrace(top_object, ray_inside, prd_radiance);

            *(&prd_radiance.result.x + colorband) *= weight;
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
        prd_radiance.result = prd_radiance.result;

    }
}
