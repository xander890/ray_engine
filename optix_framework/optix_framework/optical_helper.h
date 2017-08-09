#pragma once

#include <optix_world.h>
#include "math_helpers.h"

static __host__ __device__ __inline__ optix::float3 refract2(const optix::float3 &in, const optix::float3 & n, float eta)
{
    float c = optix::dot(in, n);
    return eta * (c * n - in) - n * sqrt(1 - eta * eta * (1 - c * c));
}

static __host__ __device__ __inline__ optix::float3 refract2(const optix::float3 &in, const optix::float3 & n, float n1, float n2)
{
    float eta = n1/n2;
	return refract2(in,n,eta);
}

static __host__ __device__ __inline__ optix::float2 fresnelAmplitudeReflectance(const optix::float3 &in, const optix::float3 &n, float nin, float ntr)
{
    float cosin = optix::dot(in,n);
    optix::float3 refr = refract2(in,n,nin,ntr);
    float costr = optix::dot(refr,-n);

    float r_s = (nin * cosin - ntr * costr) / (nin * cosin + ntr * costr);
    float r_p = (nin * costr - ntr * cosin) / (nin * costr + ntr * cosin);
    //std::cout << abs(r_s) << " ¤ " << abs(r_p) << std::endl;
    return optix::make_float2(r_s,r_p);
}


static __host__ __device__ __inline__ optix::float2 fresnelPowerReflectance(const optix::float3 &in, const optix::float3 &n, float n1, float n2)
{
    optix::float2 r = fresnelAmplitudeReflectance(in,n,n1,n2);
    return r * r;
}


static __host__ __device__ __inline__ optix::float2 fresnelAmplitudeTransmittance(const optix::float3 &in, const optix::float3 &n, float n1, float n2)
{
    float cosin = optix::dot(in,n);
    optix::float3 refr = refract2(in,n,n1,n2);
    float costr = optix::dot(refr,-n);
    //std::cout << n1 *  sqrt(1 - cosin * cosin) << " = " << n2 * sqrt(1 - costr * costr) << std::endl;
    float t_s = (2 * n1 * cosin) / (n1 * cosin + n2 * costr);
    float t_p = (2 * n1 * cosin) / (n1 * costr + n2 * cosin);
    //std::cout << abs(t_s) << " # " << abs(t_p) << std::endl;
    return optix::make_float2(t_s,t_p);
}


static __host__ __device__ __inline__ optix::float2 fresnelPowerTransmittance(const optix::float3 &in, const optix::float3 &n, float n1, float n2)
{
    float cosin = optix::dot(in,n);
    optix::float3 refr = refract2(in,n,n1,n2);
    float costr = optix::dot(refr,-n);

    optix::float2 t = fresnelAmplitudeTransmittance(in,n,n1,n2);

    return ((n2 * costr) / (n1 * cosin)) * (t * t);
}

static __host__ __device__ __inline__ float fresnel_R(const optix::float3& in, const optix::float3& n, float n1, float n2)
{
    optix::float2 R = fresnelPowerReflectance(in,n,n1,n2);
	float RR = 0.5f * (R.x + R.y);
    return RR;
}

static __host__ __device__ __inline__ optix::float3 fresnel_complex_R(float cos_theta, const optix::float3& eta_sq, const optix::float3& kappa_sq)
{
	if (cos_theta < 1e-6)
		return optix::make_float3(1.0f);
	float cos_theta_sqr = cos_theta*cos_theta;
	float sin_theta_sqr = 1.0f - cos_theta_sqr;
	float tan_theta_sqr = sin_theta_sqr/cos_theta_sqr;

  optix::float3 z_real = eta_sq - kappa_sq - sin_theta_sqr;
  optix::float3 z_imag = 4.0f*eta_sq*kappa_sq;
  optix::float3 abs_z = sqrt(z_real*z_real + z_imag*z_imag);
  optix::float3 two_a = sqrt(2.0f*(abs_z + z_real));

  optix::float3 c1 = abs_z + cos_theta_sqr;
  optix::float3 c2 = two_a*cos_theta;
  optix::float3 R_s = (c1 - c2)/(c1 + c2);


  c1 = abs_z + sin_theta_sqr*tan_theta_sqr;
  c2 = two_a*sin_theta_sqr/cos_theta;
  optix::float3 R_p = R_s*(c1 - c2)/(c1 + c2);
  return (R_s + R_p)*0.5f;
}

static __host__ __device__ __inline__ optix::float3 fresnel_complex_R(const optix::float3& in, const optix::float3& n, const optix::float3& eta_sq, const optix::float3& kappa_sq)
{
	float d = optix::dot(in, n);
    return fresnel_complex_R(d, eta_sq, kappa_sq);
}

__host__ __device__ __inline__ float fresnel_R(float cos_theta_i, float cos_theta_t, float eta)
{
  float a = eta*cos_theta_i;
  float b = eta*cos_theta_t;
  float r_s = (cos_theta_i - b)/(cos_theta_i + b);
  float r_p = (a - cos_theta_t)/(a + cos_theta_t);
  return (r_s*r_s + r_p*r_p)*0.5f;
}

__host__ __device__ __inline__ float fresnel_R(float cos_theta, float eta)
{  
  float sin_theta_t_sqr = 1.0f/(eta*eta)*(1.0f - cos_theta*cos_theta);
  if(sin_theta_t_sqr >= 1.0f) return 1.0f;
  float cos_theta_t = sqrt(1.0f - sin_theta_t_sqr);
  return fresnel_R(cos_theta, cos_theta_t, eta);
}

static __host__ __device__ __inline__ float fresnel_T(const optix::float3& in, const optix::float3& n, float n1, float n2)
{
    optix::float2 T = fresnelPowerTransmittance(in,n,n1,n2);
    float TT = 0.5f * (T.x + T.y);
    return TT;
}

static __host__ float C_1(float ni)
{
    float ni_sqr = ni*ni;
    float ni_p4 = ni_sqr*ni_sqr;
    float c;
    if(ni < 1.0f)
    {
        c = + 0.919317f
            - 3.4793f  * ni
            + 6.75335f * ni_sqr
            - 7.80989f * ni_sqr * ni
            + 4.98554f * ni_p4
            - 1.36881f * ni_p4 * ni;
    }
    else
    {
        c = - 9.23372f
            + 22.2272f  * ni
            - 20.9292f  * ni_sqr
            + 10.2291f  * ni_sqr * ni
            - 2.54396f  * ni_p4
            + 0.254913f * ni_p4 * ni;
    }
    return c * 0.5f;
}

static __host__ float C_phi(float ni)
{

    return 0.25f * (1.0f - 2.0f * C_1(ni));
}


static __host__ float C_2(float ni)
{
    float ni_sqr = ni*ni;
    float ni_p4 = ni_sqr*ni_sqr;
    float c;
    if(ni < 1.0f)
    {
        c = + 0.828421f
            - 2.62051f  * ni
            + 3.36231f  * ni_sqr
            - 1.95284f  * ni_sqr * ni
            + 0.236494f * ni_p4
            + 0.145787f * ni_p4 * ni;
    }
    else
    {
        c = - 1641.1f
            + 135.926f / (ni_sqr * ni)
            - 656.175f / ni_sqr
            + 1376.53f / ni
            + 1213.67f * ni
            - 568.556f * ni_sqr
            + 164.798f * ni_sqr * ni
            - 27.0181f * ni_p4
            + 1.91826f * ni_p4 * ni;
    }
    return c * (1.0f/3.0f);
}

static __host__ float C_E(float ni)
{
    return 0.5f * (1.0f - 3.0f * C_2(ni));
}

