#pragma once
#include "device_common.h"
#include "host_device_common.h"
#include "math_utils.h"
#include "optics_utils.h"

#define IMPROVED_ENUM_NAME NormalDistribution
#define IMPROVED_ENUM_LIST ENUMITEM_VALUE(BECKMANN,0) ENUMITEM_VALUE(GGX,1)
#include "improved_enum.inc"

_fn float positive_characteristic(const float arg)
{
    return arg > 0.0f? 1.0f : 0.0f;
}

_fn float beckmann(const optix::float3 & m, const optix::float3 & n, const float alpha)
{
    const float cos_theta = dot(m,n);
	const float cos_theta_sqr = cos_theta * cos_theta;
	float tan_theta_sqr = (1.0f - cos_theta_sqr) / cos_theta_sqr;
    const float alpha_sq = alpha*alpha;
    float cos_theta_pow_4 = cos_theta_sqr * cos_theta_sqr;
    return positive_characteristic(cos_theta) * expf(-tan_theta_sqr /alpha_sq) / (M_PIf * alpha_sq * cos_theta_pow_4);
}

_fn float beckmann_G1(const optix::float3 & v, const optix::float3 & m, const optix::float3 & n, const float alpha)
{
    const float cos_theta_v = dot(v,n);
    const float cos_theta_m = dot(m,n);
    const float a = 1.0f / (alpha * tanf(acosf(cos_theta_v)));
    const float g1 = 2.0f / (1 + erff(a) + 1.0f/(a*sqrtf(M_PIf)) * expf(-a*a));
    return positive_characteristic(cos_theta_m / cos_theta_v) * g1;
}

_fn float beckmann_G1_approx(const optix::float3 & v, const optix::float3 & m, const optix::float3 & n, const float alpha)
{
    const float cos_theta_v = dot(v,n);
    const float cos_theta_m = dot(m,n);
    const float a = 1.0f / (alpha * tanf(acosf(cos_theta_v)));
    const float g1 = a < 1.6f? (3.535f*a + 2.181f*a*a)/(1.0f + 2.276f*a + 2.577f*a*a) : 1.0f;
    return positive_characteristic(cos_theta_m / cos_theta_v) * g1;
}

_fn optix::float3 importance_sample_beckmann(const optix::float2& rand, const optix::float3 & n, const float alpha)
{
    const float theta_m = atanf(-alpha*alpha*logf(1.0f - rand.x));
    const float phi_m = 2.0f * M_PIf * rand.y;
    optix::float3 m = spherical_to_cartesian(theta_m, phi_m);
    rotate_to_normal(n,m);
    return m;
}

_fn float beckmann_anisotropic(const float cos_theta, const float alpha_x, const float alpha_y)
{
	const float cos_theta_sqr = cos_theta * cos_theta;
	const float sin_theta_sqr = 1.0f - cos_theta_sqr;
	const float tan_theta_sqr = sin_theta_sqr / cos_theta_sqr;
	const float corrected_aniso_term = cos_theta_sqr / (alpha_x*alpha_x) + sin_theta_sqr / (alpha_y*alpha_y);

	float cos_theta_pow_4 = cos_theta * cos_theta;
	cos_theta_pow_4 *= cos_theta_pow_4;
	return positive_characteristic(cos_theta) * expf(-tan_theta_sqr * corrected_aniso_term) / (M_PIf * alpha_x * alpha_y * cos_theta_pow_4);
}

_fn float beckmann_anisotropic(const optix::float3 & m, const optix::float3 & n, const float alpha_x, const float alpha_y)
{
	return beckmann_anisotropic(dot(m, n), alpha_x, alpha_y);
}

_fn optix::float3 importance_sample_beckmann_anisotropic_local(const optix::float2& rand, const float alpha_x, const float alpha_y)
{
	const float rn = sqrtf(-logf(rand.x));
	const float phi = 2.0f * M_PIf * rand.y;
	const float x = alpha_x * rn * cosf(phi);
	const float y = alpha_y * rn * sinf(phi);
	optix::float3 m = optix::normalize(optix::make_float3(x, y, 1));
	return m;
}

_fn optix::float3 importance_sample_beckmann_anisotropic(const optix::float2& rand, const optix::float3 & n, const float alpha_x, const float alpha_y)
{
	optix::float3 m = importance_sample_beckmann_anisotropic_local(rand, alpha_x, alpha_y);
	rotate_to_normal(n, m);
	return m;
}

_fn float ggx(const optix::float3 & m, const optix::float3 & n, const float alpha)
{
    const float cos_theta = dot(m,n);
    const float tan_theta = tanf(acosf(cos_theta));
    const float alpha_sq = alpha*alpha;
    float cos_theta_pow_4 = cos_theta * cos_theta;
    cos_theta_pow_4 *= cos_theta_pow_4;
    const float alpha_theta_term = (alpha_sq + tan_theta*tan_theta);
    return positive_characteristic(cos_theta) * alpha_sq / (M_PIf * cos_theta_pow_4 * alpha_theta_term*alpha_theta_term);
}

_fn float ggx_G1(float cos_theta, float alpha_g_sqr)
{
    float cos_theta_sqr = cos_theta*cos_theta;
    float tan_theta_sqr = (1.0f - cos_theta_sqr) / cos_theta_sqr;
    float G = 2.0f / (1.0f + sqrtf(1.0f + alpha_g_sqr*tan_theta_sqr));
    return G;
}

_fn float ggx_G1(optix::float3 v, optix::float3 m, optix::float3 n, float a_g)
{
    float cos_theta_v = dot(v, n);
    if (cos_theta_v * dot(v, m) <= 0)
        return 0.0f;
    return ggx_G1(cos_theta_v, a_g * a_g);
}


_fn optix::float3 importance_sample_ggx(const optix::float2& rand, const optix::float3 & n, const float alpha)
{
    const float theta_m = atanf(alpha*sqrtf(rand.x) / sqrtf(1.0f - rand.x));
    const float phi_m = 2.0f * M_PIf * rand.y;
    optix::float3 m = spherical_to_cartesian(theta_m, phi_m);
    rotate_to_normal(n,m);
    return m;
}

_fn float geometric_term_torrance_sparrow(const optix::float3& n, const optix::float3& wi, const optix::float3& wo, const optix::float3& wh)
{
    float n_dot_h = fabsf(dot(n, wh));
    float n_dot_o = fabsf(dot(n, wo));
    float n_dot_i = fabsf(dot(n, wi));
    float i_dot_o = fabsf(dot(wo, wh));
    float min_io = fminf(n_dot_o, n_dot_i);
    return fminf(1.0f, min_io * n_dot_h * 2.0f / i_dot_o);
}

_fn float torrance_sparrow_brdf(const optix::float3 & n, const optix::float3 & wi, const optix::float3 & wo, float ior, float roughness)
{
    float cos_o = dot(n, wo);
    float cos_i = dot(n, wi);
    optix::float3 half_vector = normalize(wi + wo);
    float cos_hr_wi = dot(wi, half_vector);
    float cos_hr_wo = dot(wo, half_vector);

    if (cos_hr_wo / cos_o <= 0.0f || cos_hr_wi / cos_i <= 0.0f)
        return 0.0f;

    float D = beckmann(half_vector, n, roughness);
    float G = geometric_term_torrance_sparrow(n, wi, wo, half_vector);
    float F = fresnel_R(cos_hr_wi, ior);
    float S = M_PIf * cos_o * cos_i;
    return fabsf(D * F * G / S);
}


_fn optix::float3 walter_brdf(const optix::float3 & n, const optix::float3 & wi, const optix::float3 & wo, NormalDistribution::Type normal_dist, float ior, float roughness)
{
    float cos_o = dot(n, wo);
    float cos_i = dot(n, wi);
    optix::float3 half_vector = normalize(wi + wo);
    float cos_hr_wo = dot(wo, half_vector);

    float D = 0.0f, G = 0.0f;

    if(normal_dist == NormalDistribution::BECKMANN)
    {
        D = beckmann(half_vector, n, roughness);
        G = beckmann_G1(wi, half_vector, n, roughness) * beckmann_G1(wo, half_vector, n, roughness);
    }
    else if(normal_dist == NormalDistribution::GGX)
    {
        D = ggx(half_vector, n, roughness);
        G = ggx_G1(wi, half_vector, n, roughness) * ggx_G1(wo, half_vector, n, roughness);
    }

    float F = fresnel_R(cos_hr_wo, ior);
    float S = 4.0f * cos_o * cos_i;

    optix_print("D %f F %f G %f S %f cos_o, \n", D, F, G, S);

    return optix::make_float3(fabsf(D * F * G / S));
}