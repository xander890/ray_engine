#include <device_common_data.h>
#include <random.h>
#include <sampling_helpers.h>
#include <optical_helper.h>
#include <environment_map.h>
#include "structs.h"
#include "scattering_properties.h"
#include "light.h"
#include <material_device.h>

using namespace optix;

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

// Material properties
 
rtDeclareVariable(unsigned int, N, , );
rtDeclareVariable(unsigned int, maximum_volume_steps, , );

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

__device__ __inline__ bool scatter_inside(optix::Ray& ray, int colorband, uint& t)
{
    float albedo = get_albedo(ray.origin, colorband);
    float g = get_asymmetry(ray.origin, colorband);
    float extinction = get_extinction(ray.origin, colorband);

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
    ray.ray_type = RAY_TYPE_SHADOW;
    PerRayData_shadow prd_ray;
    prd_ray.attenuation = 1.0f;

#ifdef USE_SIMILARITY

    bool stop = false;
    int counter = 0;
    bool absorbed = false;
    while (!stop)
    {
        albedo = get_albedo(ray.origin, colorband);
        g = get_asymmetry(ray.origin, colorband);
        extinction = get_extinction(ray.origin, colorband);
        float red_extinction = get_reduced_extinction(ray.origin, colorband);

        // Sample new distance
        float dist_exponent = (counter < SIMILARITY_STEPS) ? extinction : red_extinction;
        ray.tmax = -log(rnd(t)) / dist_exponent;
        prd_ray.attenuation = 1.0f;

        rtTrace(top_shadower, ray, prd_ray);

        if (prd_ray.attenuation > 0.0f)
        {
            // The shadow ray did not hit anything, i.e., still inside the volume
            // New ray origin
            ray.origin += ray.direction * ray.tmax;
            // New ray direction 
            ray.direction = (counter < SIMILARITY_STEPS) ? sample_HG(ray.direction, g, t) : sample_HG(ray.direction, t);
        }
        else break;

        absorbed = rnd(t) > albedo;
        stop = absorbed || prd_ray.attenuation <= 0.0f;

    }
    return !absorbed;
#else
    for (int i = 0; i < maximum_volume_steps; i++)
    {
        extinction = get_extinction(ray.origin, colorband);
        // Sample new distance
        ray.tmax = -log(rnd(t)) / extinction;

        rtTrace(top_shadower, ray, prd_ray);
        if (prd_ray.attenuation > 0.0f)
        {
            // The shadow ray did not hit anything, i.e., still inside the volume

            // New ray origin
            ray.origin += ray.direction * ray.tmax;

            g = get_asymmetry(ray.origin, colorband);
            // New ray direction 
            ray.direction = sample_HG(ray.direction, g, t);
        }
        else // Intersection hit
            return true;

        albedo = get_albedo(ray.origin, colorband);
        // Break if absorbed
        if (rnd(t) > albedo)
            return false;
    }
#endif
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
    float R = 1.0f;
    float sin_theta_out_sqr = n1_over_n2*n1_over_n2*(1.0f - cos_theta_in*cos_theta_in);
    float3 reflected_dir = 2.0f*cos_theta_in*normal - w_i;
    float3 refracted_dir;
    if (sin_theta_out_sqr < 1.0f)
    {
        float cos_theta_out = sqrtf(1.0f - sin_theta_out_sqr);
        R = fresnel_R(cos_theta_in, cos_theta_out, n1_over_n2);
        refracted_dir = n1_over_n2*(normal*cos_theta_in - w_i) - normal*cos_theta_out;
    }

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
        Ray ray_inside(hit_pos, dir_inside, RAY_TYPE_SHADOW, scene_epsilon, RT_DEFAULT_MAX);

        if (scatter_inside(ray_inside, colorband, t))
        {
            // Switch to radiance ray and intersect with boundary
            ray_inside.ray_type = RAY_TYPE_RADIANCE;
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
        Ray ray_outside(hit_pos, dir_outside, RAY_TYPE_RADIANCE, scene_epsilon, RT_DEFAULT_MAX);
        rtTrace(top_object, ray_outside, prd_radiance);
        prd_radiance.result = beam_T * (prd_radiance.result);

    }
}
