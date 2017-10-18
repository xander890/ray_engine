#pragma once
#include <optix_world.h>
#include "math_helpers.h"
#include "random.h"

// Sample hemisphere
static
	__host__ __device__ __inline__ optix::float3 sample_hemisphere_cosine( const optix::float2 & sample, const optix::float3 & normal )
{
	float cos_theta = sqrt(fmaxf(0.0f, sample.x));
	float phi = 2.0f * M_PIf * sample.y;
	float sin_theta = sqrt(fmaxf(0.0f,1.0f - sample.x));

	optix::float3 v = optix::make_float3(cos(phi)*sin_theta, sin(phi)*sin_theta, cos_theta);
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
	optix::float3 r = -reflect(dir_out, normal);
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
	optix::float3 dNdx = dNdP*dPdx;
	float dDNdx = dot(dDdx,N) + dot(D,dNdx);
	return dDdx - 2*(dot(D,N)*dNdx + dDNdx*N);
}

// Compute the direction ray differential for refraction
static __host__ __device__ __inline__ 
	optix::float3 differential_refract_direction(optix::float3 dPdx, optix::float3 dDdx, optix::float3 dNdP, 
	optix::float3 D, optix::float3 N, float ior, optix::float3 T)
{
	float eta;
	if(dot(D,N) > 0.f) {
		eta = ior;
		N = -N;
	} else {
		eta = 1.f / ior;
	}

	optix::float3 dNdx = dNdP*dPdx;
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

static __inline__ __device__ optix::float3 sample_hemisphere_uniform(const optix::float2 & sample, const optix::float3 & normal)
{
	optix::float3 p;
	p.z = sample.x;
	const float r = sqrtf(fmaxf(0.0f, 1.0f - p.z * p.z));
	const float phi = 2.0f * M_PIf * sample.y;
	p.x = r * cos(phi);
	p.y = r * sin(phi);
	return p;
}

static __inline__ __device__ optix::float2 sample_disk(const optix::float2 & sample, float minR = 0.0f)
{
	optix::float2 p;
	const float r = fmaxf(sqrtf(sample.x), minR);
	const float phi = 2.0f * M_PIf * sample.y;
	p.x = r * cos(phi);
	p.y = r * sin(phi);
	return p;
}

static __inline__ __device__ optix::float2 sample_disk_exponential(optix::uint & seed, float sigma, float & pdf, float & r, float & phi)
{
	optix::float2 sample = optix::make_float2(rnd(seed), rnd(seed));
	optix::float2 p;
	r = -log(sample.x) / sigma;
	phi = 2.0f * M_PIf * sample.y;
	p.x = r * cos(phi);
	p.y = r * sin(phi);
	pdf = sigma * sample.x / (2.0f* M_PIf); 
	return p;
}

static __inline__ __device__ float exponential_pdf_disk(float r, float sigma)
{
	return M_1_PIf * 0.5f * sigma * expf(-sigma * r);
}

static __inline__ __device__ optix::float2 sample_disk_exponential(optix::uint & seed, float sigma, float & pdf)
{
	float r, phi;
	return sample_disk_exponential(seed, sigma, pdf, r, phi);
}

__host__ __device__ __inline__ optix::float3 burley_scaling_factor_mfp_searchlight(const optix::float3 & albedo)
{
	optix::float3 temp = abs(albedo - optix::make_float3(0.8f));
	return optix::make_float3(1.85f) - albedo + 7.0f * temp * temp * temp;
}

__host__ __device__ __inline__ optix::float3 burley_scaling_factor_diffuse_mfp_searchlight(const optix::float3 & albedo)
{
	optix::float3 temp = albedo - optix::make_float3(0.33f);
	temp *= temp; // pow 2
	temp *= temp; // pow 4
	return optix::make_float3(3.5f) + 100.0f * temp;
}


__host__ __device__ __inline__ optix::float3 burley_scaling_factor_mfp_diffuse(const optix::float3 & albedo)
{
	optix::float3 temp = abs(albedo - optix::make_float3(0.8f));
	return optix::make_float3(1.9f) - albedo + 3.5f * temp * temp;
}