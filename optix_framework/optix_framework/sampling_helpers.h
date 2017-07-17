#pragma once

#include <optix_world.h>
#include <optix.h>
#include "math_helpers.h"

// Sample hemisphere
static
	__host__ __device__ __inline__ optix::float3 sample_hemisphere_cosine( const optix::float2 & sample, const optix::float3 & normal )
{

	float cos_theta = sqrt(fmaxf(0.0f, sample.x));
	float phi = 2.0f * M_PIf * sample.y;
  float sin_theta = sqrt(fmaxf(0.0f,1.0f - sample.x));

	float3 v = make_float3(cos(phi)*sin_theta, sin(phi)*sin_theta, cos_theta);
	rotate_to_normal(normal,v);
	return v;
}

// Sample Phong lobe relative to U, V, W frame
static
	__host__ __device__ __inline__ optix::float3 sample_phong_lobe( optix::float2 sample, float exponent, 
	optix::float3 U, optix::float3 V, optix::float3 W )
{
	const float power = expf( logf(sample.y)/(exponent+1.0f) );
	const float phi = sample.x * 2.0f * (float)M_PIf;
	const float scale = sqrtf(1.0f - power*power);

	const float x = cosf(phi)*scale;
	const float y = sinf(phi)*scale;
	const float z = power;

	return x*U + y*V + z*W;
}

// Sample Phong lobe relative to U, V, W frame
static
	__host__ __device__ __inline__ optix::float3 sample_phong_lobe( const optix::float2 &sample, float exponent, 
	const optix::float3 &U, const optix::float3 &V, const optix::float3 &W, 
	float &pdf, float &bdf_val )
{
	const float cos_theta = powf(sample.y, 1.0f/(exponent+1.0f) );

	const float phi = sample.x * 2.0f * M_PIf;
	const float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);

	const float x = cosf(phi)*sin_theta;
	const float y = sinf(phi)*sin_theta;
	const float z = cos_theta;

	const float powered_cos = powf( cos_theta, exponent );
	pdf = (exponent+1.0f) / (2.0f*M_PIf) * powered_cos;
	bdf_val = (exponent+2.0f) / (2.0f*M_PIf) * powered_cos;  

	return x*U + y*V + z*W;
}

// Get Phong lobe PDF for local frame
static
	__host__ __device__ __inline__ float get_phong_lobe_pdf( float exponent, const optix::float3 &normal, const optix::float3 &dir_out, 
	const optix::float3 &dir_in, float &bdf_val)
{  
	using namespace optix;

	float3 r = -reflect(dir_out, normal);
	const float cos_theta = fabs(dot(r, dir_in));
	const float powered_cos = powf(cos_theta, exponent );

	bdf_val = (exponent+2.0f) / (2.0f*M_PIf) * powered_cos;  
	return (exponent+1.0f) / (2.0f*M_PIf) * powered_cos;
}



// Compute the origin ray differential for transfer
static
	__host__ __device__ __inline__ optix::float3 differential_transfer_origin(optix::float3 dPdx, optix::float3 dDdx, float t, optix::float3 direction, optix::float3 normal)
{
	float dtdx = -optix::dot((dPdx + t*dDdx), normal)/optix::dot(direction, normal);
	return (dPdx + t*dDdx)+dtdx*direction;
}

// Compute the direction ray differential for a pinhole camera
static
	__host__ __device__ __inline__ optix::float3 differential_generation_direction(optix::float3 d, optix::float3 basis)
{
	float dd = optix::dot(d,d);
	return (dd*basis-optix::dot(d,basis)*d)/(dd*sqrtf(dd));
}

// Compute the direction ray differential for reflection
static
	__host__ __device__ __inline__
	optix::float3 differential_reflect_direction(optix::float3 dPdx, optix::float3 dDdx, optix::float3 dNdP, 
	optix::float3 D, optix::float3 N)
{
	using namespace optix;

	float3 dNdx = dNdP*dPdx;
	float dDNdx = dot(dDdx,N) + dot(D,dNdx);
	return dDdx - 2*(dot(D,N)*dNdx + dDNdx*N);
}

// Compute the direction ray differential for refraction
static __host__ __device__ __inline__ 
	optix::float3 differential_refract_direction(optix::float3 dPdx, optix::float3 dDdx, optix::float3 dNdP, 
	optix::float3 D, optix::float3 N, float ior, optix::float3 T)
{
	using namespace optix;

	float eta;
	if(dot(D,N) > 0.f) {
		eta = ior;
		N = -N;
	} else {
		eta = 1.f / ior;
	}

	float3 dNdx = dNdP*dPdx;
	float mu = eta*dot(D,N)-dot(T,N);
	float TN = -sqrtf(1-eta*eta*(1-dot(D,N)*dot(D,N)));
	float dDNdx = dot(dDdx,N) + dot(D,dNdx);
	float dmudx = (eta - (eta*eta*dot(D,N))/TN)*dDNdx;
	return eta*dDdx - (mu*dNdx+dmudx*N);
}

// zeta1, zeta2 are two random uniform iid in [0,1], the function gives a uniform distributed point inside a triangle defined by v0,v1,v2
static __host__ __device__ __inline__ optix::float3 sample_point_triangle(float zeta1, float zeta2, optix::float3 v0, optix::float3 v1, optix::float3 v2)
{
	// As in Osada, Robert: Shape Distributions
	zeta1 = sqrt(zeta1);
	return (1-zeta1) * v0 + zeta1 * (1-zeta2) * v1 + zeta1 * zeta2 * v2;
}
