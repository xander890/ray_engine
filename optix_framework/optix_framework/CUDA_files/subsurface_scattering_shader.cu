// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011
#include <device_common.h>
#include <math_utils.h>
#include <random_device.h>
#include <bssrdf_device.h>

#include <optics_utils.h>
#include <structs.h>
#include <ray_tracing_utils.h>
#include <scattering_properties.h>
#include <material_device.h>

using namespace optix;


// Standard ray variables
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload,);
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload,);

// SS properties

// Variables for shading
rtDeclareVariable(BufPtr<PositionSample>, sampling_output_buffer, ,);
rtDeclareVariable(float3, shading_normal, attribute shading_normal,);
rtDeclareVariable(float2, texcoord, attribute texcoord,);
rtDeclareVariable(int, exclude_backfaces, , );

// Any hit program for shadows
RT_PROGRAM void any_hit_shadow()
{
    // this material is opaque, so it fully attenuates all shadow rays
    prd_shadow.attenuation = 0.0f;
    rtTerminateRay();
}

// Closest hit program for Lambertian shading using the basic light as a directional source
RT_PROGRAM void shade()
{
    if (prd_radiance.depth > max_depth)
    {
        prd_radiance.result = make_float3(0.0f);
        return;
    }

    TEASampler * sampler = prd_radiance.sampler;

    float3 no = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 xo = ray.origin + t_hit*ray.direction;
    float3 wo = -ray.direction;
    const MaterialDataCommon material = get_material(xo);
    const ScatteringMaterialProperties& props = material.scattering_properties;
    float relative_ior = dot(material.index_of_refraction, optix::make_float3(1)) / 3.0f;
    float recip_ior = 1.0f / relative_ior;
    float reflect_xi = sampler->next1D();
    prd_radiance.result = make_float3(0.0f);

    float3 beam_T = make_float3(1.0f);
    float cos_theta_o = dot(wo, no);
    bool inside = cos_theta_o < 0.0f;
    if (inside)
    {
        beam_T = get_beam_transmittance(t_hit, props);
        float prob = (beam_T.x + beam_T.y + beam_T.z) / 3.0f;
        if (sampler->next1D() >= prob) return;
        beam_T /= prob;
        recip_ior = relative_ior;
        cos_theta_o = -cos_theta_o;
        no = -no;
    }

    float3 wt;
    float R;
    refract(wo, no, recip_ior, wt, R);

    if (reflect_xi >= R)
    {
        PerRayData_radiance prd_refracted = prepare_new_pt_payload(prd_radiance);

        Ray refracted(xo, wt, RayType::RADIANCE, scene_epsilon);
        rtTrace(top_object, refracted, prd_refracted);
        prd_radiance.result += prd_refracted.result * beam_T;

        if (!inside)
        {
            float chosen_transport_rr = dot(props.transport, make_float3(0.33333f));
            float3 accumulate = make_float3(0.0f);
            int N = sampling_output_buffer.size();

            for (int i = 0; i < N; ++i)
            {
                PositionSample &sample = sampling_output_buffer[i];

                // compute direction of the transmitted light
                const float3 &wi = sample.dir;

                // compute contribution if sample is non-zero
                if (dot(sample.L, sample.L) > 0.0f)
                {
                    // Russian roulette
                    float dist = length(xo - sample.pos);
                    float exp_term = expf(-dist * chosen_transport_rr);
                    if (prd_radiance.sampler->next1D() < exp_term)
                    {
                        if(exclude_backfaces == 1 && dot(no, sample.normal) < 0.0f)
                            continue;

                        BSSRDFGeometry geometry;
                        geometry.xi = sample.pos;
                        geometry.ni = sample.normal;
                        geometry.wi = wi;
                        geometry.xo = xo;
                        geometry.no = no;
                        geometry.wo = wo;
                        accumulate += sample.L *
                                      bssrdf(geometry, recip_ior, material, BSSRDFFlags::EXCLUDE_OUTGOING_FRESNEL, *prd_radiance.sampler) /
                                      exp_term;
                        optix_print("L %f %f %f, exp = %f\n", sample.L.x, sample.L.y, sample.L.z, 1 / exp_term);
                    }
                }

            }
            prd_radiance.result += accumulate / (float) N;
        }
    }

    // Trace reflected ray
    if (reflect_xi < R)
    {
        float3 wr = -reflect(wo, no);
        PerRayData_radiance prd_reflected = prepare_new_pt_payload(prd_radiance);
        Ray reflected(xo, wr, RayType::RADIANCE, scene_epsilon);
        rtTrace(top_object, reflected, prd_reflected);
        prd_radiance.result += prd_reflected.result;
    }
}
