#include "sky_model.h"
#include "folders.h"
#include "host_device_common.h"

void SkyModel::update_data()
{
    if (dot(north, up) != 0.0f)
        throw std::runtime_error("North and up are not perpendicular.");

    perez_data.A = make_float3(0.1787f * turbidity - 1.4630f, -0.0193f * turbidity - 0.2592f, -0.0167f * turbidity - 0.2608f);
    perez_data.B = make_float3(-0.3554f * turbidity + 0.4275f, -0.0665f * turbidity + 0.0008f, -0.0950f * turbidity + 0.0092f);
    perez_data.C = make_float3(-0.0227f * turbidity + 5.3251f, -0.0004f * turbidity + 0.2125f, -0.0079f * turbidity + 0.2102f);
    perez_data.D = make_float3(0.1206f * turbidity - 2.5771f, -0.0641f * turbidity - 0.8989f, -0.0441f * turbidity - 1.6537f);
    perez_data.E = make_float3(-0.0670f * turbidity + 0.3703f, -0.0033f * turbidity + 0.0452f, -0.0109f * turbidity + 0.0529f);

    solar_coords = get_solar_coordinates();

    sun_position = get_sun_position(solar_coords);
    float sun_theta = solar_coords.x;
    float sun_theta_2 = sun_theta * sun_theta;
    float sun_theta_3 = sun_theta_2 * sun_theta;

    const float xi = (4.0f / 9.0f - turbidity / 120.0f) * (M_PIf - 2.0f * sun_theta);

    // Zenith luminance in Yxy
    float3 zenith = make_float3(0.0f);
    zenith.x = ((4.0453f * turbidity - 4.9710f) * tan(xi) - 0.2155f * turbidity + 2.4192f) * 1000.0f;
    zenith.y = turbidity * turbidity * (0.00166f*sun_theta_3 - 0.00375f*sun_theta_2 + 0.00209f*sun_theta) +
        turbidity * (-0.02903f*sun_theta_3 + 0.06377f*sun_theta_2 - 0.03202f*sun_theta + 0.00394f) +
        (0.11693f*sun_theta_3 - 0.21196f*sun_theta_2 + 0.06052f*sun_theta + 0.25886f);
    zenith.z = turbidity * turbidity * (0.00275f*sun_theta_3 - 0.00610f*sun_theta_2 + 0.00317f*sun_theta) +
        turbidity * (-0.04214f*sun_theta_3 + 0.08970f*sun_theta_2 - 0.04153f*sun_theta + 0.00516f) +
        (0.15346f*sun_theta_3 - 0.26756f*sun_theta_2 + 0.06670f*sun_theta + 0.26688f);

    cos_sun_theta = cos(sun_theta);
    sky_factor = zenith / perez_model(1.0f, sun_theta, cos_sun_theta, perez_data);
    sun_color = get_sun_color() * 0.1f;
}

void SkyModel::init(optix::Context & ctx)
{
    MissProgram::init(ctx);
    update_data();
}

SkyModel::~SkyModel(void)
{

}

float SkyModel::get_solar_declination()
{
	static const float DECL_CONST = 2.0f/368.0f * M_PIf; 
	return 0.4093f * sin((float)hour * DECL_CONST);
}

float2 SkyModel::get_solar_coordinates()
{
	float time = (float)hour * M_PIf / 12.0f;
	float delta = get_solar_declination();
	float sin_delta = sin(delta);
	float cos_delta = cos(delta);
	float sin_lat = sin(deg2rad(latitude));
	float cos_lat = cos(deg2rad(latitude));
	float sin_time = sin(time);
	float cos_time = cos(time);
	float theta_s = M_PI_2f - asin(sin_lat * sin_delta - cos_lat * cos_delta * cos_time);
	float phi_s = atan(- cos_delta * sin_time / (cos_lat * sin_delta - sin_lat * cos_delta * cos_time));
	return make_float2(theta_s,phi_s);
}


float3 SkyModel::get_sun_position(float2 & coords)
{
	float sin_theta = sin(coords.x);
	float cos_theta = cos(coords.x);
	float sin_phi = sin(coords.y);
	float cos_phi = cos(coords.y);
	float3 sun_pos = make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

	rotate_to_normal(up, sun_pos);

	return sun_pos;
}

float SkyModel::calculate_absorption( float sun_theta, float m, float lambda, float turbidity, float k_o, float k_wa ) 
{
	float alpha = 1.3f;                             // wavelength exponent
	float beta  = 0.04608f * turbidity - 0.04586f;  // turbidity coefficient
	float ell   = 0.35f;                            // ozone at NTP (cm)
	float w     = 2.0f;                             // precipitable water vapor (cm)

	float rayleigh_air   = exp( -0.008735f * m * pow( lambda, -4.08f ) );
	float aerosol        = exp( -beta * m * pow( lambda, -alpha ) );
	float ozone          = k_o  > 0.0f ? exp( -k_o*ell*m )                                         : 1.0f;
	float water_vapor    = k_wa > 0.0f ? exp( -0.2385f*k_wa*w*m/pow( 1.0f + 20.07f*k_wa*w*m, 0.45f ) ) : 1.0f;

	return rayleigh_air*aerosol*ozone*water_vapor;
}



float3 SkyModel::get_sky_color(float3 v)
{
	return sky_color(0,v,sun_position,up,sky_factor,sun_color,perez_data);
}

float3 SkyModel::get_sun_color()
{

	float decl_angle = 93.885f - solar_coords.x * 180.0f / M_PIf;
	float optical_mass = 1.0f / (cos_sun_theta + 0.15f * pow(decl_angle, -1.253f));

	float3 color = make_float3(0.0f);
	for(int i = 0; i < 38; i++)
	{
		PreethamData data_elem = data[i];
		float radiance = data_elem.sun_spectral_radiance * 10000.0f / 1000.0f; //in the right quantity now
		radiance *= calculate_absorption(solar_coords.x,optical_mass,data_elem.wavelength, turbidity, data_elem.k_o, data_elem.k_wa);
		color.x += radiance * cie_table[i][1] * 10.0f;
		color.y += radiance * cie_table[i][2] * 10.0f;
		color.z += radiance * cie_table[i][3] * 10.0f;
	}
	return  XYZ2rgb(683.0f * color) / 1000.0f;
}

bool SkyModel::get_miss_program(unsigned ray_type, optix::Context& ctx, optix::Program& program)
{
    const char * prog = ray_type ==  RayType::SHADOW ? "miss_shadow" : "miss";
    program = ctx->createProgramFromPTXFile(get_path_ptx("sky_model_background.cu"), prog);
    return true;
}


void SkyModel::get_directional_light(SingularLightData & light) const
{
	light.direction = -sun_position;
	light.emission = sun_color * 6.87e-5f;
	light.casts_shadow = 1;
}


const float SkyModel::cie_table[38][4] =
{
	{380.f, 0.0002f, 0.0000f, 0.0007f},
	{390.f, 0.0024f, 0.0003f, 0.0105f},
	{400.f, 0.0191f, 0.0020f, 0.0860f},
	{410.f, 0.0847f, 0.0088f, 0.3894f},
	{420.f, 0.2045f, 0.0214f, 0.9725f},

	{430.f, 0.3147f, 0.0387f, 1.5535f},
	{440.f, 0.3837f, 0.0621f, 1.9673f},
	{450.f, 0.3707f, 0.0895f, 1.9948f},
	{460.f, 0.3023f, 0.1282f, 1.7454f},
	{470.f, 0.1956f, 0.1852f, 1.3176f},

	{480.f, 0.0805f, 0.2536f, 0.7721f},
	{490.f, 0.0162f, 0.3391f, 0.4153f},
	{500.f, 0.0038f, 0.4608f, 0.2185f},
	{510.f, 0.0375f, 0.6067f, 0.1120f},
	{520.f, 0.1177f, 0.7618f, 0.0607f},

	{530.f, 0.2365f, 0.8752f, 0.0305f},
	{540.f, 0.3768f, 0.9620f, 0.0137f},
	{550.f, 0.5298f, 0.9918f, 0.0040f},
	{560.f, 0.7052f, 0.9973f, 0.0000f},
	{570.f, 0.8787f, 0.9556f, 0.0000f},

	{580.f, 1.0142f, 0.8689f, 0.0000f},
	{590.f, 1.1185f, 0.7774f, 0.0000f},
	{600.f, 1.1240f, 0.6583f, 0.0000f},
	{610.f, 1.0305f, 0.5280f, 0.0000f},
	{620.f, 0.8563f, 0.3981f, 0.0000f},

	{630.f, 0.6475f, 0.2835f, 0.0000f},
	{640.f, 0.4316f, 0.1798f, 0.0000f},
	{650.f, 0.2683f, 0.1076f, 0.0000f},
	{660.f, 0.1526f, 0.0603f, 0.0000f},
	{670.f, 0.0813f, 0.0318f, 0.0000f},

	{680.f, 0.0409f, 0.0159f, 0.0000f},
	{690.f, 0.0199f, 0.0077f, 0.0000f},
	{700.f, 0.0096f, 0.0037f, 0.0000f},
	{710.f, 0.0046f, 0.0018f, 0.0000f},
	{720.f, 0.0022f, 0.0008f, 0.0000f},

	{730.f, 0.0010f, 0.0004f, 0.0000f},
	{740.f, 0.0005f, 0.0002f, 0.0000f},
	{750.f, 0.0003f, 0.0001f, 0.0000f}
};

const SkyModel::PreethamData SkyModel::data[] = 
{
	{0.38f, 1655.9f, -1.f, -1.f},
	{0.39f, 1623.37f, -1.f, -1.f},
	{0.4f, 2112.75f, -1.f, -1.f},
	{0.41f, 2588.82f, -1.f, -1.f},
	{0.42f, 2582.91f, -1.f, -1.f},
	{0.43f, 2423.23f, -1.f, -1.f},
	{0.44f, 2676.05f, -1.f, -1.f},
	{0.45f, 2965.83f, 0.003f, -1.f},
	{0.46f, 3054.54f, 0.006f, -1.f},
	{0.47f, 3005.75f, 0.009f, -1.f},

	{0.48f, 3066.37f, 0.014f, -1.f},
	{0.49f, 2883.04f, 0.021f, -1.f},
	{0.5f, 2871.21f, 0.03f, -1.f},
	{0.51f, 2782.5f, 0.04f, -1.f},
	{0.52f, 2710.06f, 0.048f, -1.f},
	{0.53f, 2723.36f, 0.063f, -1.f},
	{0.54f, 2636.13f, 0.075f, -1.f},
	{0.55f, 2550.38f, 0.085f, -1.f},
	{0.56f, 2506.02f, 0.103f, -1.f},
	{0.57f, 2531.16f, 0.12f, -1.f},

	{0.58f, 2535.59f, 0.12f, -1.f},
	{0.59f, 2513.42f, 0.115f, -1.f},
	{0.6f, 2463.15f, 0.125f, -1.f},
	{0.61f, 2417.32f, 0.12f, -1.f},
	{0.62f, 2368.53f, 0.105f, -1.f},
	{0.63f, 2321.21f, 0.09f, -1.f},
	{0.64f, 2282.77f, 0.079f, -1.f},
	{0.65f, 2233.98f, 0.067f, -1.f},
	{0.66f, 2197.02f, 0.057f, -1.f},
	{0.67f, 2152.67f, 0.048f, -1.f},

	{0.68f, 2109.79f, 0.036f, -1.f},
	{0.69f, 2072.83f, 0.028f, 0.028f},
	{0.7f, 2024.04f, 0.023f, 0.023f},
	{0.71f, 1987.08f, 0.018f, 0.018f},
	{0.72f, 1942.72f, 0.014f, 0.014f},
	{0.73f, 1907.24f, 0.011f, 0.011f},
	{0.74f, 1862.89f, 0.01f, 0.01f},
	{0.75f, 1825.92f, 0.009f, 0.009f}
};

void SkyModel::set_into_gpu(optix::Context & context)
{
    MissProgram::set_into_gpu(context);
	context["perez_model_data"]->setUserData(sizeof(PerezData), &perez_data);
	context["sun_position"]->setFloat(sun_position);
	context["up_vector"]->setFloat(up);
	context["sky_factor"]->setFloat(sky_factor);
	context["sun_color"]->setFloat(sun_color);
}
