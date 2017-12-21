#include <device_common_data.h>
#include <math_helpers.h>
#include <random.h>
#include <optical_helper.h>
#include <ray_trace_helpers.h>
#include <scattering_properties.h>
#include <sampling_helpers.h>

__device__ __forceinline__ float map_interval(float point, optix::float2 interval_from, optix::float2 interval_to)
{
    float map_01 = (point - interval_from.x)/(interval_from.y - interval_from.x);
    return interval_to.x + map_01 * (interval_to.y - interval_to.x);
}

__device__ __forceinline__ void sample_neural_network(
        const float3 & xo,          // The points hit by the camera ray.
        const float3 & no,          // The normal at the point.
        const float3 & wo,          // The incoming ray direction.
        const MaterialDataCommon & material,  // Material properties.
        TEASampler * sampler,       // A rng.
	    float3 & x_tangent,                // The candidate point 
        float & integration_factor, // An factor that will be multiplied into the final result. For inverse pdfs. 
        float3 & proposed_wi)   
{   
    // Gathering scattering parameters.
    // For now only red channel.
    float albedo = material.scattering_properties.albedo.x;
    float extinction = material.scattering_properties.extinction.x;
    float eta = material.relative_ior;
    float g = material.scattering_properties.meancosine.x;

    float cos_theta_i = dot(wo, no);
    float theta_i = acosf(cos_theta_i);        

    // Sampling random inputs for NN.
    float x1 = sampler->next1D();
    float x2 = sampler->next1D();
    float x3 = sampler->next1D();
    float x4 = sampler->next1D();

    optix::float4 hypernetwork_parameters = make_float4(eta,g,albedo,theta_i);
    optix::float4 cdfnetwork_parameters = make_float4(x1,x2,x3,x4);

    // Sample neural network here!
    float y1 = 0, y2 = 0, y3 = 0, y4 = 0; 

    // Also here sample normalization constant
    float bssrdf_integral = 0.0f;
    integration_factor *= bssrdf_integral;

    float r = map_interval(y1, optix::make_float2(0,1), optix::make_float2(0.01f, 10.0f));           
    float theta_s = map_interval(y2, optix::make_float2(0,1), optix::make_float2(0.0f, M_PIf));           
    float theta_o = map_interval(y3, optix::make_float2(0,1), optix::make_float2(0.0f, M_PIf/2));           
    float phi_o = map_interval(y4, optix::make_float2(0,1), optix::make_float2(0.0f, 2*M_PIf));           

    // The radius is expressed in mean free paths, so we renormalize it.
    r /= extinction;

    // Pick a random side for theta_s
    float zeta = sampler->next1D();
    theta_s *= (zeta > 0.5f)? -1 : 1;
    integration_factor *= 2;

    // Note that the tangent vector has to be aligned to wo in order to get a consistent framae for theta_s.
    float3 to = normalize(wo - cos_theta_i * no);
    float3 bo = cross(to, no);
    x_tangent = xo + r * cosf(theta_s) * to + r * sinf(theta_s) * bo;

    proposed_wi = spherical_to_cartesian(theta_o, phi_o);

}
