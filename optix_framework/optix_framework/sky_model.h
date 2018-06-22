#pragma once
#include "light_common.h"
#include "math_utils.h"
#include "color_utils.h"
#include <optix_math.h>
#pragma once

struct PerezData
{
	optix::float3 A,B,C,D,E;
};

__constant__ const float SKY_SCALE = 0.03f;

static _fn optix::float3 perez_model(float cos_theta, float gamma, float cos_gamma, PerezData & data)
{
	const optix::float3 one = optix::make_float3(1.0f);
	return (one + data.A * exp(data.B / cos_theta)) * (one + data.C * exp(data.D * gamma) + data.E * cos_gamma * cos_gamma);
}

static _fn optix::float3 sky_color(int ray_depth, optix::float3 & v, optix::float3& sun_position, optix::float3 & up, optix::float3 & sky_factor, optix::float3 & sun_color, PerezData & data)
{
	float cos_gamma = dot(v, sun_position);
	float cos_theta = dot(v,up);
	if (cos_theta < 0.0f)	//Lower part of the sky
		cos_theta = -cos_theta;
	//if (cos_gamma > 0.9999f && ray_depth == 0) // actual sun
	//	return sun_color;
	float gamma = acos(cos_gamma);
	optix::float3 lum = sky_factor * perez_model(cos_theta, gamma, cos_gamma,data);
	return Yxy2rgb(lum) * SKY_SCALE / 1000.0f;
}

#ifndef __CUDA_ARCH__
#include "optix_serialize_utils.h"
#include <cmath>
#include "miss_program.h"

class SkyModel : public MissProgram
{
public:
	SkyModel(optix::Context & ctx, optix::float3 up = optix::make_float3(0,0,1), optix::float3 north = optix::make_float3(1,0,0)) : MissProgram(ctx), up(up), north(north) {  }
	~SkyModel(void);
	optix::float3 get_sky_color(optix::float3 v);
	void get_directional_light(SingularLightData & light) const;

    virtual void init() override;
    virtual void load() override;
	virtual bool on_draw() override { return false; }

private:

	int day = 12;
	int hour = 15;
	float latitude = 45.0f;
	optix::float3 up;
	optix::float3 north;
	float turbidity = 2.0f;
	PerezData perez_data;
	optix::float3 sky_factor;
	float cos_sun_theta;
	optix::float3 sun_position;
	optix::float2 solar_coords;
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
    void update_data();

	float get_solar_declination();
	optix::float2 get_solar_coordinates();
	optix::float3 get_sun_position(optix::float2 & coords);
	float calculate_absorption( float sun_theta, float m, float lambda, float turbidity, float k_o, float k_wa );
	optix::float3 get_sun_color();

private:
    virtual bool get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program) override;

	static void load_and_construct(cereal::XMLInputArchiveOptix & archive, cereal::construct<SkyModel>& construct)
	{
		optix::Context ctx = archive.get_context();
		construct(ctx);
		archive(cereal::virtual_base_class<MissProgram>(construct.ptr()),
			cereal::make_nvp("up", construct->up),
			cereal::make_nvp("north", construct->north),
			cereal::make_nvp("day", construct->day),
			cereal::make_nvp("hour", construct->hour),
			cereal::make_nvp("latitude", construct->latitude),
			cereal::make_nvp("turbidity", construct->turbidity)
		);
	}

    friend class cereal::access;
    template<class Archive>
    void save(Archive & archive) const
    {
        archive(cereal::virtual_base_class<MissProgram>(this), CEREAL_NVP(up), CEREAL_NVP(north), CEREAL_NVP(day), CEREAL_NVP(hour), CEREAL_NVP(latitude) , CEREAL_NVP(turbidity));
    }
};

CEREAL_CLASS_VERSION(SkyModel, 0)
CEREAL_REGISTER_TYPE(SkyModel)
CEREAL_REGISTER_POLYMORPHIC_RELATION(MissProgram, SkyModel)
#endif // !__CUDA_ARCH__
