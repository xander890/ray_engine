#ifndef structs_h__
#define structs_h__

#pragma once
#include <optix_world.h>

struct ScatteringMaterialProperties
{
	 // base parameters
     float indexOfRefraction; //The usual assumption is that this can be intercheangeably the material ior or the ratio between it and air (ior = 1)
     optix::float3 absorption;
     optix::float3 scattering;
     optix::float3 meancosine;

    // derived parameters
     optix::float3 D;
     optix::float3 reducedExtinction;
     optix::float3 transport;
     optix::float3 reducedScattering;
     float C_s;
     float C_s_inv;
     float C_E;
     float A;
     optix::float3 de;
     optix::float3 reducedAlbedo;
     optix::float3 extinction;
     optix::float3 three_D;
     optix::float3 rev_D;
     float global_coeff;
     optix::float3 two_a_de;
     optix::float3 two_de;
     optix::float3 one_over_three_ext;
     optix::float3 de_sqr;
     float iorsq;
     float min_transport;
};

struct MPMLMedium
{
	std::string	  name;
	optix::float3 ior_real;
	optix::float3 ior_imag;
	optix::float3 emission;
	optix::float3 extinction;    
	optix::float3 scattering;
	optix::float3 absorption;
	optix::float3 asymmetry;
	optix::float3 albedo;
	optix::float3 reduced_scattering;
	optix::float3 reduced_extinction;    
	optix::float3 reduced_albedo;
};

/*
Flag arithmetics:
flags &= ~(FLAG)     -		clear
flags |= FLAG		 -		set
flags ^= FLAG		 -      toggle
flags &  FLAG        -		check
*/
namespace RayFlags
{
	const int HIT_DIFFUSE_SURFACE		= 1 << 0;
	const int USE_EMISSION				= 1 << 1;
	const int DEBUG_PIXEL				= 1 << 2;
	const int NO_DIRECT_ON_FIRST_HIT	= 1 << 3;
	const int RETURN_FIRST_HIT_POSITION = 1 << 4;
	const int RETURN_LAMBERTIAN_HIT_UV =  1 << 5;
}
// Payload for radiance ray type
struct PerRayData_radiance
{
  optix::float3 result;
  float importance;
  unsigned int depth;
  int colorband;
  unsigned int seed;
  unsigned int flags;
};


struct PerRayData_cache
{
	optix::float3 vertex;
	optix::float3 normal;
	PerRayData_radiance radiance;
};

// Payload for shadow ray type
struct PerRayData_shadow
{
  float attenuation;
  optix::float3 emission;
};

struct PositionSample
{
  optix::float3 pos;
  optix::float3 dir;
  optix::float3 normal;
  optix::float3 L;
};
#endif // structs_h__