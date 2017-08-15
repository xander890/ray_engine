#include <device_common_data.h>
#include <random.h>
#include <sampling_helpers.h>
#include <optical_helper.h>
#include <environment_map.h>
#include "structs.h"
#include "scattering_properties.h"
#include "light.h"
#include <material.h>

using namespace optix;

// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

// Variables for shading
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

// Material properties
rtDeclareVariable(MaterialDataCommon, material, , ); 
rtDeclareVariable(unsigned int, N, , );
rtDeclareVariable(float3, glass_abs, , );


//#define USE_SIMILARITY
#define SIMILARITY_STEPS 10

#ifdef USE_SIMILARITY
__device__ __inline__ bool scatter_inside(optix::Ray& ray, float g, float extinction, float red_extinction, float albedo, uint& t)
#else
__device__ __inline__ bool scatter_inside(optix::Ray& ray, float g, float extinction, float albedo, uint& t)
#endif
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
    ray.ray_type = shadow_ray_type;
    PerRayData_shadow prd_ray;
    prd_ray.attenuation = 1.0f;

#ifdef USE_SIMILARITY

    bool stop = false;
    int counter = 0;
    while (!stop)
    {
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
    for (;;)
    {
        // Sample new distance
        ray.tmax = -log(rnd(t)) / extinction;

        rtTrace(top_shadower, ray, prd_ray);
        if (prd_ray.attenuation > 0.0f)
        {
            // The shadow ray did not hit anything, i.e., still inside the volume

            // New ray origin
            ray.origin += ray.direction * ray.tmax;

            // New ray direction 
            ray.direction = sample_HG(ray.direction, g, t);
        }
        else // Intersection hit
            return true;

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
    ScatteringMaterialProperties& props = material.scattering_properties;
    float n1_over_n2 = 1.0f / props.relative_ior;
    float cos_theta_in = dot(normal, w_i);
    float3 beam_T = make_float3(1.0f);
    uint& t = prd_radiance.seed;

    // Russian roulette with absorption if arrived from dense medium
    bool inside = cos_theta_in < 0.0f;
    if (inside)
    {
        n1_over_n2 = props.relative_ior;
        normal = -normal;
        cos_theta_in = -cos_theta_in;
    }
    else if (props.relative_ior < 1.0f)
    {
        beam_T = expf(-t_hit*glass_abs);
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
        float color_albedo = *(&props.albedo.x + colorband);
        float color_extinction = *(&props.extinction.x + colorband);
        float g = *(&props.meancosine.x + colorband);

        // Trace inside, i.e., reflect from inside or refract from outside
        float3 dir_inside = inside ? reflected_dir : refracted_dir;
        Ray ray_inside(hit_pos, dir_inside, shadow_ray_type, scene_epsilon, RT_DEFAULT_MAX);
#ifdef USE_SIMILARITY
        float reduced_color_extinction = *(&props.reducedExtinction.x + colorband);
        if (scatter_inside(ray_inside, g, color_extinction, reduced_color_extinction, color_albedo, t))
#else
        if (scatter_inside(ray_inside, g, color_extinction, color_albedo, t))
#endif
        {
            // Switch to radiance ray and intersect with boundary
            ray_inside.ray_type = radiance_ray_type;
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
        Ray ray_outside(hit_pos, dir_outside, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);
        rtTrace(top_object, ray_outside, prd_radiance);
        prd_radiance.result = beam_T * (prd_radiance.result);

    }
}
