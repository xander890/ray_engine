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
rtDeclareVariable(float, empirical_bssrdf_correction,,);
rtDeclareVariable(unsigned int, empirical_bssrdf_interpolation,,);


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

__device__ __forceinline__ int get_index_closest(int parameter_index, const float value)
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


__device__ __forceinline__ int get_index_lower(int parameter_index, const float value)
{
    auto values = empirical_bssrdf_parameters.buffers[parameter_index];
    float candidate_value = values[0];
    int candidate_index = 0;
    for(int i = 0; i < values.size(); i++)
    {
        if(values[i] < value && fabsf(values[i] - value) < fabsf(candidate_value - value))
        {
            candidate_value = values[i];
            candidate_index = i;
        }
    }
    return candidate_index;
}

__device__ __forceinline__ int get_index_upper(int parameter_index, const float value)
{
    auto values = empirical_bssrdf_parameters.buffers[parameter_index];
    float candidate_value = values[0];
    int candidate_index = 0;
    for(int i = 0; i < values.size(); i++)
    {
        if(values[i] >= value && fabsf(values[i] - value) < fabsf(candidate_value - value))
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
        closest_index = get_index_closest(j, value);
        float closest_param_value = empirical_bssrdf_parameters.buffers[j][closest_index];
        index_function[j] = closest_index;
        //optix_print("SIZE %d/%d closet %f - asked %f\n", closest_index, empirical_bssrdf_parameters_size[j], closest_param_value, value);
    }
    int index = ravel<N>(index_function, empirical_bssrdf_parameters_size);
    if( index > empirical_buffer_size) {
        printf("Index is out of bounds! %d / %d\n", index, empirical_buffer_size);
        return 0;
    }
    //optix_print("INDEX %d -- %e\n", index, empirical_buffer.buffers[slice][index]);
    return empirical_buffer.buffers[slice][index];
}


template<int N>
__device__ __forceinline__ float interpolate_bssrdf_linear(float values[N], int slice)
{
    float interpolated = 0.0f;
    for(int i = 0; i < (1 << N); i++)
    {
        int index_function[N];
        float factor = 1;
        const char * v[5] = {"theta_s", "r", "theta_i", "phi_o", "theta_o"};
        for(int j = 0; j < N; j++)
        {
            unsigned int bit = (i >> j) & 0x01;
            float value = values[j];
            int closest_index_lower = get_index_lower(j, value);
            int closest_index_upper = get_index_upper(j, value);
            if(closest_index_lower == closest_index_upper)
            {
                // Outside of the domain or hitting exactly the correct point.
                index_function[j] = closest_index_lower;
                // factor *= 1 is implicit.
            }
            else
            {
                float closest_param_value = empirical_bssrdf_parameters.buffers[j][closest_index_lower];
                float closest_param_value_upper = empirical_bssrdf_parameters.buffers[j][closest_index_upper];
                float value_normalized =
                        (value - closest_param_value) / (closest_param_value_upper - closest_param_value);
                if (launch_index.x == 0 && launch_index.y == 0)
                    printf("%s (%f) %f %d %d\n", v[j], value, value_normalized, closest_index_lower,
                           closest_index_upper);
                value_normalized = clamp(value_normalized, 0.0f, 1.0f);
                index_function[j] = (bit == 0) ? closest_index_lower : closest_index_upper;
                factor *= (bit == 0) ? 1 - value_normalized : value_normalized;
            }
        }
        int index = ravel<N>(index_function, empirical_bssrdf_parameters_size);
        interpolated += empirical_buffer.buffers[slice][index] * factor;
    }
    return interpolated;
}



__forceinline__ __device__ optix::float3 eval_empbssrdf(const BSSRDFGeometry geometry, const float n1_over_n2,
                                                        const MaterialDataCommon material, unsigned int flags, TEASampler & sampler)
{
    optix_print("EMPIRICAL\n");
    float theta_i, r, theta_s, theta_o, phi_o;
    empirical_bssrdf_get_geometry(geometry, theta_i,  r, theta_s, theta_o, phi_o);

    optix::float3 S;
    for(int i = 0; i < 3; i++)
    {
        float extinction = optix::get_channel(i, material.scattering_properties.extinction);
        float r_s = r * extinction;
        r_s = clamp(r_s, 0.01f, 10.0f);
        float values[5] = {theta_s, r_s, theta_i, theta_o, phi_o};
        //optix_print("theta_s %f\n", theta_s);
        optix_print("r %f (ext %f - %f)\n", r_s, extinction, r);
        //optix_print("theta_i %f\n", theta_i);
        //optix_print("theta_o %f\n", theta_o);
        //optix_print("phi_o %f\n", phi_o);

        float SS = 0;
        if(empirical_bssrdf_interpolation == 0)
            SS = interpolate_bssrdf_nearest<5>(values,i) * empirical_bssrdf_correction;
        else
            SS = interpolate_bssrdf_linear<5>(values,i) * empirical_bssrdf_correction;
        optix::get_channel(i, S) = SS * extinction * extinction;
    }

    optix::float3 w21;
    float R21;
    bool include_fresnel_out = (flags &= BSSRDFFlags::EXCLUDE_OUTGOING_FRESNEL) == 0;

    refract(-geometry.wo, -geometry.no, 1.0/n1_over_n2, w21, R21);

    float F = 1;//1.0f/(1.0f - R21);
    return S * F;
}


#endif



