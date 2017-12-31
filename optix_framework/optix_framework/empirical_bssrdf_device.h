#pragma once
#include "host_device_common.h"
#include  "bssrdf_properties.h"
#include  "material_device.h"
#include <device_common_data.h>
#include <scattering_properties.h>
//#include <bitset>
#include "empirical_bssrdf_utils.h"
#include "optix_helpers.h"

#ifdef INCLUDE_PROGRAMS_ONLY
rtDeclareVariable(rtCallableProgramId<optix::float3(const BSSRDFGeometry, const float, const MaterialDataCommon, unsigned int, TEASampler *)>, empirical_bssrdf, , );
#else

rtDeclareVariable(EmpiricalDataBuffer, empirical_buffer, , );
rtDeclareVariable(int, empirical_buffer_size, , );
rtDeclareVariable(EmpiricalParameterBuffer, empirical_bssrdf_parameters, , );
rtDeclareVariable(BufPtr<int>, empirical_bssrdf_parameters_size, , );

template<int N>
__host__ __device__ __forceinline__ int ravel(int idx[N], BufPtr<int>& size)
{
	size_t id = idx[0];
	for (int i = 1; i < N; i++)
	{
		id = id * size[i] + idx[i];
	}
	return id;
}

template<int N>
__host__ __device__ __forceinline__ void unravel(const size_t& idx, BufPtr<int>& size, int * res)
{
	size_t index = idx;
    for (int i = N - 1; i >= 0; i--)
	{
		res[i] = index % size[i];
		index = index / size[i];
	}
}

__device__ __forceinline__ int get_index(int parameter_index, const float value)
{
    auto values = empirical_bssrdf_parameters.buffers[parameter_index];
    float candidate_value = -1e9f;
    int candidate_index = 0;
    for(int i = 0; i < values.size(); i++)
    {
        if(fabsf(values[i] - value) < fabsf(candidate_value - value))
        {
            candidate_value = values[i];
            candidate_index = i;
        }
    }
    return candidate_index;
}

template<int N>
__device__ __forceinline__ float interpolate_bssrdf_nearest(float values[N], int slice)
{

    int index_function[N];
    for(int j = 0; j < N; j++)
    {
        float value = values[j];
        int closest_index = 0;
        closest_index = get_index(j, value);
        float closest_param_value = empirical_bssrdf_parameters.buffers[j][closest_index];
        index_function[j] = closest_index;
        optix_print("SIZE %d/%d closet %f - asked %f\n", closest_index, empirical_bssrdf_parameters_size[j], closest_param_value, value);
    }
    int index = ravel<N>(index_function, empirical_bssrdf_parameters_size);
    if( index > empirical_buffer_size) {
        optix_print("Index is out of bounds! %d / %d\n", index, empirical_buffer_size);
        return 0;
    }
    optix_print("INDEX %d -- %e\n", index, empirical_buffer.buffers[slice][index]);
    return empirical_buffer.buffers[slice][index];
}

/*
template<int N>
__device__ __forceinline__ float interpolate_bssrdf(float values[N], const rtBufferId<float> & slice)
{
    float interpolated = 0.0f;
    for(int i = 0; i < (1 << N); i++)
    {
        std::bitset<N> bits(i);
        int index_function[N];
        float factor = 1;
        for(int j = 0; j < N; j++)
        {
            unsigned int bit = bits[j];
            float value = values[j];
            int closest_index;
            float closest_param_value;
            get_index(empirical_bssrdf_parameters[j], value, closest_index, closest_param_value);
            float closes_param_value_upper = empirical_bssrdf_parameters[j][closest_index + 1];
            float value_normalized = (value - closest_param_value) / (closest_param_value_upper - closest_param_value);
            index_function[j] = (bit == 0)? closest_index : closest_index + 1;
            factor *= (bit == 0)? 1 - value_normalized : value_normalized;
        }
        int index = ravel<N>(index_function, empirical_bssrdf_parameters_size);
        interpolated += slice[index] * factor;
    }
    return interpolated;
}*/

__forceinline__ __device__ optix::float3 eval_empbssrdf(const BSSRDFGeometry geometry, const float recip_ior,
                                                        const MaterialDataCommon material, unsigned int flags = BSSRDFFlags::NO_FLAGS, TEASampler * sampler = nullptr)
{
    optix_print("EMPIRICAL\n");

    float cos_theta_i = dot(geometry.wi, geometry.ni);
    float theta_i = acosf(cos_theta_i);

    optix::float3 x = geometry.xo - geometry.xi;
    optix::float3 x_norm = normalize(x);
    float cos_theta_o = dot(geometry.no, geometry.wo);
    optix::float3 x_bar = -normalize(geometry.wi - cos_theta_i * geometry.ni);

    if(fabs(theta_i) <= 1e-6f)
    {
        x_bar = x_norm;
    }

    optix::float3 z_bar = normalize(cross(geometry.ni, x_bar));
    float theta_s = -atan2(dot(z_bar, x_norm),dot(x_bar, x_norm));

    // theta_s mirroring.
    if(theta_s < 0) {
        theta_s = abs(theta_s);
        z_bar = -z_bar;
    }

    optix::float3 xo_bar = normalize(geometry.wo - cos_theta_o * geometry.no);
    float theta_o = acosf(cos_theta_o);
    float phi_o = atan2f(dot(z_bar,xo_bar), dot(x_bar,xo_bar));

    phi_o = normalize_angle(phi_o);

            optix_assert(theta_i >= 0 && theta_i <= M_PIf/2);
            optix_assert(theta_s >= 0 && theta_s <= M_PIf);
            optix_assert(theta_o >= 0 && theta_o <= M_PIf/2);
            optix_assert(phi_o >= 0 &&  phi_o < 2*M_PIf);

    optix::float3 S;
    for(int i = 0; i < 3; i++)
    {
        float r = optix::length(x) * optix::get_channel(i, material.scattering_properties.extinction);
        r = clamp(r, 0.01f, 10.0f);
        float values[5] = {theta_s, r, theta_i, theta_o, phi_o};
        optix_print("theta_s %f\n", theta_s);
        optix_print("r %f\n", r);
        optix_print("theta_i %f\n", theta_i);
        optix_print("theta_o %f\n", theta_o);
        optix_print("phi_o %f\n", phi_o);

        optix::get_channel(i, S) = interpolate_bssrdf_nearest<5>(values,i);
    }
    optix_print("S: %e %e %e\n", S.x, S.y, S.z);

    optix::float3 w21;
    float R21;
    bool include_fresnel_out = (flags &= BSSRDFFlags::EXCLUDE_OUTGOING_FRESNEL) == 0;

    refract(geometry.wo, geometry.no, recip_ior, w21, R21);
    float F = include_fresnel_out?  1.0f : 1.0f/(1.0f - R21);
    return S * F;
}


#endif



