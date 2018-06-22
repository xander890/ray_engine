// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011
#include "device_common.h"
#include "optics_utils.h"
#include "brdf_common.h"
#include "material_common.h"
#include "sampler_device.h"
#include "microfacet_utils.h"

rtDeclareVariable(float, ridge_angle, , ) = 20.0f;

_fn optix::float3 qr_plane_projection(const optix::float3& w, const optix::float3& u)
{
	optix::float3 w_p = normalize(w - dot(w, u) * u);
	return w_p;
}
_fn void qr_create_onb(const optix::float3& n, optix::float3& u, optix::float3& v, const float& t_x, const float& t_y)
{
	//create_onb(n, u, v);
	//return;
	const int span = 10;
	const int vshift = int(t_y* span + 1) % 2;
	bool toggle = int(t_x * span + vshift) % 2 == 0;
	if (toggle) {
		create_onb(n, u, v);
	}
	else {
		create_onb(n, v, u);
	}
}
_fn optix::uint qr_ridge_side(const optix::float3& w_p, const optix::float3& v, const float& theta_p, const float& slope, float& weight, const float& z)
{
	//ridge_side=1 if w_p hit the slope, ridge_side = 0 if w_p hit the edge
	optix::uint ridge_side = dot(w_p, v) > 0;
	if (ridge_side == 0) {
		float prob = fminf(1.0f, tan(theta_p) * tan(slope));
		if (z < prob) {
			//hit edge
			ridge_side = 0;
		}
		else {
			//hit slope
			ridge_side = 1;
		}
	}
	return ridge_side;
}
_fn void qr_get_weight_factors(const optix::float3& w_i, const optix::float3& n, const optix::float3& m, const optix::float3& w_p, const float& ior1_over_ior2, const optix::uint& ridge_side, float& F_r, float& den)
{
	F_r = fmaxf(0.0f, fresnel_R(fmaxf(0.0f, (dot(w_i, m))), ior1_over_ior2));
	if (ridge_side) {
		den = fabsf(dot(w_i, n) * dot(m, n) / dot(w_i, m));
	}
	else {
		den = fabsf(dot(w_i, n));
		den = 1.0f;
	}
}
_fn float ridged_G(const optix::float3& w_p, const optix::float3& n, const optix::float3& m, const float& slope, const optix::uint& ridge_side) {
	float G;
	optix::uint chi;
	if (ridge_side == 1) {
		//hit the slope side
		chi = dot(w_p, n) > 0.0f &&  dot(w_p, m) > 0.0f &&  dot(n, m) > 0.0f;
		float theta = acos(dot(w_p, m));
		//float shadowed_slope = sin(theta + slope) / cos(theta) * sin(slope);
		float shadowed_slope = tan(theta)*tan(slope);
		G = optix::clamp(1.0f - fminf(1.0f, shadowed_slope), 0.0f, 1.0f);
	}
	else {
		//hit the vertical edge
		chi = dot(w_p, n) > 0.0f &&  dot(w_p, m) > 0.0f;
		float theta = acos(dot(w_p, n));
		G = optix::clamp(1.0f / (tan(theta) * tan(slope)), 0.0f, 1.0f);
		G = 0.0f;
	}
	return G * chi;
}
//Beckmann distribution normal sampling in object coordinates
_fn void qr_sample_normal(const optix::float3& n, const optix::float3& u, const optix::float3& v,
	optix::float3 &sampled_n, const float slope, const optix::uint& ridge_side, const float roughness, const float z)
{
	if (ridge_side == 0) {
		//if hit the vertical edge
		sampled_n = -v;
		return;
	}
	//if hit the slope
	float phi = 0;
	float width = roughness;
	float tan_theta = -width*width * log(1 - z);
	float theta = atan(sqrtf(tan_theta)) + slope;
	//theta = slope;
	optix::clamp(theta, -M_PI_2f, M_PI_2f);
	float costheta = cosf(theta);
	float sintheta = sinf(theta);
	sampled_n = optix::make_float3(sintheta * sin(phi), sintheta * cos(phi), costheta);
	//transform the sampled_normal coordinates from world to object
	sampled_n = normalize(u * sampled_n.x + v * sampled_n.y + n * sampled_n.z);
}
//Beckmann anisotropic distribution normal sampling in object coordinates
_fn void qr_sample_anisotropic_beckmann_normal(const optix::float3& ideal_m, const optix::float3& u,
	const optix::float3& v, optix::float3& m, const float a_u, const float a_v,
	const float z1, const float z2, const optix::uint ridge_side)
{
	if (ridge_side == 0) {
		//if hit the vertical edge
		m = -v;
		return;
	}
	optix::float3 u_m = normalize(cross(ideal_m, v));
	optix::float3 v_m = normalize(cross(ideal_m, u_m));
	optix::float3 local_n = importance_sample_beckmann_anisotropic_local(optix::make_float2(z1,z2), a_u, a_v);
	m = normalize(u_m * local_n.x + v_m * local_n.y + ideal_m * local_n.z);
}
//Beckmann anisotropic distribution normal evaluation in object coordinates
_fn float qr_eval_anisotropic_beckmann_D(const optix::float3& ideal_m, const optix::float3& u,
	const optix::float3& v, optix::float3& m, const float a_u, const float a_v, const optix::uint ridge_side)
{
	float D = 1.0f;
	if (ridge_side == 0) {
		//if hit the vertical edge
		return D;
	}
	optix::float3 u_m = normalize(cross(ideal_m, v));
	optix::float3 v_m = normalize(cross(ideal_m, u_m));
	optix::float3 local_n = optix::make_float3(dot(m, u_m), dot(m, v_m), dot(m, ideal_m));
	D = beckmann_anisotropic(dot(local_n, optix::make_float3(0,0,1)), a_u, a_v);
	return D;
}
//Gaussian random number
_fn float qr_sample_gaussian(const float& std_dev, const float& mean, const float& z1, const float& z2)
{
	return  sqrtf(-2.0f * log(z1)) * cosf(2.0f * M_PI *z2)*std_dev + mean;
}
//Gaussian  distribution evaluation
_fn float qr_evaluate_gaussian(const float x, const float& std_dev, const float& mean)
{
	float std_dev_sqr = std_dev*std_dev;
	float x_mean_sqr = (x - mean)*(x - mean);
	return 1.0f / sqrtf(2.0f * M_PI * std_dev_sqr) * exp(-x_mean_sqr / (2.0f * std_dev_sqr));
}

//Gaussian sample normal
//_fn void qr_sample_gaussian_normal(const optix::float3& n, const optix::float3& u, const optix::float3& v,
//	optix::float3 &sampled_n, const float slope, const optix::uint& ridge_side, const float roughness, optix::uint& seed)
//{
//	if (ridge_side == 0) {
//		//if hit the vertical edge
//		sampled_n = -v;
//		return;
//	}
//	//if hit the slope
//	float width = roughness;
//	float theta;
//	float phi = qr_sample_gaussian(width / 4.0f, 0, rnd(seed), rnd(seed));
//	float tan_theta = qr_sample_gaussian(width, slope, rnd(seed), rnd(seed));
//	//theta = slope;
//	optix::clamp(theta, -M_PI_2f, M_PI_2f);
//	optix::clamp(phi, -M_PI_2f, M_PI_2f);
//	float cos_theta = cosf(theta);
//	float sin_theta = sinf(theta);
//	float cos_phi = cosf(phi);
//	float sin_phi = sinf(phi);
//	optix::float3 theta_n = optix::make_float3(sin_theta * sinf(0), sin_theta * cosf(0), cos_theta);
//	optix::float3 phi_n = optix::make_float3(sin_phi * cosf(0), sin_phi * sinf(0), cos_phi);
//	sampled_n = normalize(theta_n + phi_n);
//	//transform the sampled_normal coordinates from world to object
//	sampled_n = normalize(u * sampled_n.x + v * sampled_n.y + n * sampled_n.z);
//}

//evaluate gaussian distribution
_fn float qr_evaluate_gaussian_D(const optix::float3& n, const optix::float3& u, const optix::float3& v,
	optix::float3 &sampled_n, const float slope, const float roughness, const optix::uint ridge_side)
{
	if (ridge_side == 0) {
		//if hit the vertical edge
		return 1.0f;
	}
	//if hit the slope
	float width = roughness;
	optix::float3 vn_proj = qr_plane_projection(sampled_n, u);
	optix::float3 un_proj = qr_plane_projection(sampled_n, v);
	float cos_theta = fminf(1.0f, dot(n, vn_proj));
	float cos_phi = fminf(1.0f, dot(un_proj, n));
	float theta = acos(abs(cos_theta));
	float phi = acos(abs(cos_phi));
	if (dot(v, vn_proj) < 0.0f) {
		theta = -theta;
	}
	if (dot(u, un_proj) < 0.0) {
		phi = -phi;
	}
	float D_theta = qr_evaluate_gaussian(theta, width, slope);
	float D_phi = qr_evaluate_gaussian(phi, width / 4.0f, 0.0f);
	float D = D_theta * D_phi;
	return D;
}
//evaluate beckmann distribution
_fn float qr_evaluate_beckmann_D(const optix::float3& n, const optix::float3& u,
	const optix::float3& v, const optix::float3& m, float slope, const float new_slope,
	const float roughness, const optix::uint ridge_side)
{
	if (ridge_side == 0) {
		//if hit the vertical edge
		return 1.0f;
	}
	//if hit the slope
	float D = 0.0f;
	float threshold = 1e-6f;
	if (dot(n, m) < 0.0f || abs(dot(m, u)) > threshold)
	{
		return D;
	}
	float rough_sqr = roughness * roughness;
	float theta = new_slope - slope;
	float tan_theta = tanf(theta);
	float tan_theta_sqr = tan_theta * tan_theta;
	float cos_theta = cosf(theta);
	float cos_theta_four = cos_theta * cos_theta * cos_theta * cos_theta;
	D = exp(-tan_theta_sqr / rough_sqr) / (M_PI * rough_sqr * cos_theta_four);
	return D;
}

_fn void importance_sample_qr_brdf(BRDFGeometry & geometry,
	const MaterialDataCommon& material, TEASampler & sampler, optix::float3 & new_direction, optix::float3 & importance_sampled_brdf)
{
	//hit point variables
	optix::float3 hit_point = ray.origin + t_hit * ray.direction;
	optix::float3 normal = geometry.n;
	optix::float3 ffnormal = optix::faceforward(normal, -ray.direction, normal);
	optix::float3 w_i = geometry.wo;
	optix::float3 rho_d = make_float3(optix::rtTex2D<optix::float4>(material.diffuse_map, geometry.texcoord.x, geometry.texcoord.y));

	const float relative_ior = dot(material.index_of_refraction, optix::make_float3(1.0f)) / 3.0f;
	float ior1_over_ior2 = 1.0f / relative_ior;
	float slope = deg2rad(ridge_angle);
	float roughness = material.roughness;
	optix::uint ridge_side = 1;
	//used normal
	optix::float3 n = ffnormal;
	float cos_i_n = dot(w_i, n);
	/*Texturized the plane in order to have chessboard pattern with different ridges orientation
	the ridge are going along the v direction*/
	optix::float3 u, v;
	create_onb(n, u, v);
	u = rotate_around(u, n, deg2rad(material.anisotropy_angle));
	v = rotate_around(v, n, deg2rad(material.anisotropy_angle));

	optix::float3 ideal_m = normalize(u * 0.0f + v * sin(slope) + n *cos(slope));
	//incoming direction projected on the plane n-v
	optix::float3 w_p_i = qr_plane_projection(w_i, u);
	float theta_p = acos(dot(w_p_i, n));
	//if 1 is on the slope, 0 is on the vertical edge

	float z1 = sampler.next1D();
	//ridge_side = qr_ridge_side(w_p_i, v, theta_p, slope,  weight, z1);
	float z2 = sampler.next1D();
	optix::float3 m;

	float a_v = roughness;
	float a_u = roughness / 4.0f;
	qr_sample_anisotropic_beckmann_normal(ideal_m, u, v, m, a_u, a_v, z1, z2, ridge_side);
	//qr_sample_normal(n, u, v, m, slope, ridge_side, roughness, z2);
	//qr_sample_gaussian_normal(n, u, v, m, slope, ridge_side, roughness, seed);
	float new_slope = acos(dot(n, m));
	if (dot(m, v) < 0.0f)
	{
		new_slope = -new_slope;
	}
	float cos_i_m = dot(w_i, m);
	float cos_m_n = dot(m, n);
	float G_i = ridged_G(w_p_i, n, m, new_slope, ridge_side);
	//outgoing direction
	optix::float3 w_o = optix::reflect(ray.direction, m);
	optix::float3 w_p_o = qr_plane_projection(w_o, u);
	float G_o = ridged_G(w_p_o, n, m, new_slope, ridge_side);
	//qr_get_weight_factors(w_i, n, m, w_p_i, ior1_over_ior2, ridge_side, F_r, den);
	float F_r = fmaxf(0.0f, fresnel_R(fmaxf(0.0f, (dot(w_i, m))), ior1_over_ior2));
	//float den = fabsf(dot(w_i, n) * dot(m, n) / dot(w_i, m));
	float correction_factor = fabsf(dot(w_i, m)) / fmaxf(1.0e-4, fabsf((dot(w_i, n) * dot(m, n))));
	float spec_weight = 1.0f;
	spec_weight *= G_i * G_o * F_r * correction_factor;
	float diff_weight = 1.0f;
	float F_t_i_m = 1.0f - fresnel_R(fabsf(cos_i_m), ior1_over_ior2);
	float F_t_o_m = F_t_i_m;
	diff_weight *= F_t_i_m * F_t_o_m * G_i * G_o * correction_factor * M_1_PI;
	//reflected ray
	new_direction = w_o;
	geometry.wi = w_o;

	if (G_i > 0.0f) {
		importance_sampled_brdf = optix::make_float3(spec_weight) + rho_d * diff_weight;
	}
	else
	{
		importance_sampled_brdf = optix::make_float3(0);
	}
}