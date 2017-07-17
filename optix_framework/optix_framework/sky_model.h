#ifndef sky_model_h__
#define sky_model_h__

#include "directional_light.h"

#include <optix_math.h>
#include "math_helpers.h"
#include "color_helpers.h"


using namespace optix;
#pragma once

struct PerezData
{
	float3 A,B,C,D,E;
};

__constant__ const float SKY_SCALE = 0.03f;

static __inline__ __device__ __host__ float3 perez_model(float cos_theta, float gamma, float cos_gamma, PerezData & data)
{
	const float3 one = make_float3(1.0f);
	return (one + data.A * exp(data.B / cos_theta)) * (one + data.C * exp(data.D * gamma) + data.E * cos_gamma * cos_gamma);
}

static __inline__ __device__ __host__ float3 sky_color(int ray_depth, float3 & v, float3& sun_position, float3 & up, float3 & sky_factor, float3 & sun_color, PerezData & data)
{
	float cos_gamma = dot(v, sun_position);
	float cos_theta = dot(v,up);
	if (cos_theta < 0.0f)	//Lower part of the sky
		cos_theta = -cos_theta;
	//if (cos_gamma > 0.9999f && ray_depth == 0) // actual sun
	//	return sun_color;
	float gamma = acos(cos_gamma);
	float3 lum = sky_factor * perez_model(cos_theta, gamma, cos_gamma,data);
	return Yxy2rgb(lum) * SKY_SCALE / 1000.0f;
}

#ifndef __CUDA_ARCH__
#include "parameter_parser.h"
#define _USE_MATH_DEFINES
#include <cmath>

class SkyModel
{
public:
	SkyModel(float3 up, float3 north) : up(up), north(north) {  }
	~SkyModel(void);
	float3 get_sky_color(float3 v);
	void get_directional_light(DirectionalLight & light);
	void load_data_on_GPU(optix::Context & context);
	void init();

private:

	int day;
	int hour;
	float latitude;
	float3 up;
	float3 north;
	float turbidity;
	PerezData perez_data;
	float3 sky_factor;
	float cos_sun_theta;
	float3 sun_position;
	float2 solar_coords;
	optix::float3 sun_color;

	struct PreethamData
	{
		float wavelength;
		float sun_spectral_radiance;
		float k_o;
		float k_wa;
	};

	static const float cie_table[38][4];   
	static const PreethamData data[38]; 
	

	float get_solar_declination();
	float2 get_solar_coordinates();
	float3 get_sun_position(float2 & coords);
	float calculate_absorption( float sun_theta, float m, float lambda, float turbidity, float k_o, float k_wa );
	float3 get_sun_color();

};
#endif // !__CUDA_ARCH__

#endif // sky_model_h__
