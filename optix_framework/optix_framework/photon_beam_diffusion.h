#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>
#include <optix_helpers.h>
#include <optical_helper.h>
#include <phase_function.h>
#include <bssrdf_properties.h>
#include "material.h"

_fn float bdp_bssrdf(float d_r, float z_r, optix::float4 props, optix::float4 C) { // Better dipole if z_r = 1/sigma_t_p and d_r = sqrt(z_r^2 + r^2)
	float sigma_s = props.x, sigma_a = props.y, g = props.z, A = C.w;
	float sigma_s_p = sigma_s*(1.0 - g);
	float sigma_t_p = sigma_s_p + sigma_a;
	float alpha_p = sigma_s_p / sigma_t_p;
	float D = 1.0 / (3.0*sigma_t_p);
	D *= (sigma_a*2.0 + sigma_s_p) / sigma_t_p; // Grosjeans approximation
	float d_e = 4.0*A*D;
	float sigma_tr = sqrt(sigma_a / D);
	float d_r_sqr = d_r*d_r;
	float d_v_sqr = d_r_sqr + 2.0*z_r*d_e + d_e*d_e; // d_r^2 - z_r^2 + z_v^2
	float d_v = sqrt(d_v_sqr);
	float tr_r = sigma_tr*d_r;
	float S_r = z_r*(1.0 + tr_r) / (d_r_sqr*d_r);
	float T_r = exp(-tr_r);
	S_r *= T_r;
	float tr_v = sigma_tr*d_v;
	float S_v = (z_r + d_e)*(1.0 + tr_v) / (d_v_sqr*d_v);
	float T_v = exp(-tr_v);
	S_v *= T_v;
	float phi = (T_r / d_r - T_v / d_v) / D;
	float S_d = phi*C.y + (S_r + S_v)*C.z;
	return alpha_p*M_1_4PIPIf*C.x*S_d;
}

_fn float single_diffuse(float t, float d_r, optix::float3 w_i, optix::float3 w_o, optix::float3 n_o, optix::float4 props) {
	float sigma_s = props.x, sigma_a = props.y, g = props.z;
	float sigma_t = sigma_s + sigma_a;
	float cos_theta_o = abs(dot(w_o, n_o));
	return sigma_s*phase_HG(dot(w_i, w_o), g)*expf(-sigma_t*(t + d_r))*cos_theta_o / (d_r*d_r);
}
_fn float pbd_bssrdf(optix::float3 xi, optix::float3 ni, optix::float3 wt, optix::float3 xo, optix::float3 no, optix::float4 props, optix::float4 C) {
	const float N = 5.0f;
	float sigma_s = props.x, sigma_a = props.y, g = props.z;
	float sigma_t = sigma_s + sigma_a;
	float sigma_s_p = sigma_s*(1.0 - g);
	float sigma_t_p = sigma_s_p + sigma_a;
	float alpha_p = sigma_s_p / sigma_t_p;
	optix::float3 x = xo - xi;
	float a = 0.9 / sigma_t_p;
	float b = 1.1 / sigma_t_p;
	float w_exp = clamp((length(x) - a) / (b - a), 0.0, 1.0);
	float w_equ = 1.0 - w_exp;
	float cos_theta_t = dot(wt, -ni);
	float Delta = dot(x, wt);                         // signed distance to perpendicular
	float h = length(Delta*wt - x);                   // perpendicular distance to beam
	float theta_a = atan2(-Delta, h);
	float theta_b = 0.5*M_PIf;
	float S_d = 0.0;
	for (int i = 1.0; i <= N; ++i) {
		float xi_i = (i - 0.5f) / N;                       // deterministic regular sequence
		float t_i = -log(1.0 - xi_i) / sigma_t_p;         // exponential sampling
		optix::float3 xr_xo = x - t_i*wt;
		float d_r = length(xr_xo);
		float z_r = t_i*cos_theta_t;
		float kappa = 1.0 - exp(-2.0*sigma_t*(d_r + t_i));
		float pdf_exp = sigma_t_p*exp(-sigma_t_p*t_i);
		float Q = alpha_p*pdf_exp;
		float f = bdp_bssrdf(d_r, z_r, props, C)*Q*kappa;
		float s = single_diffuse(t_i, d_r, wt, normalize(xr_xo), no, props)*M_1_PIf*C.x*kappa;
		float theta_j = lerp(theta_a, theta_b, xi_i);
		float t_j = h*tan(theta_j) + Delta;             // equiangular sampling
		float t_equ = t_i - Delta;                      // multiple importance sampling
		float pdf_equ = h / ((theta_b - theta_a)*(h*h + t_equ*t_equ));
		S_d += (f + s)*w_exp / (w_exp*pdf_exp + w_equ*pdf_equ);  // exponential sampling part (t_i)
		t_equ = t_j - Delta;
		pdf_equ = h / ((theta_b - theta_a)*(h*h + t_equ*t_equ));
		pdf_exp = sigma_t_p*exp(-sigma_t_p*t_j);
		xr_xo = x - t_j*wt;
		d_r = length(xr_xo);
		z_r = t_j*cos_theta_t;
		kappa = 1.0 - exp(-2.0*sigma_t*(d_r + t_j));
		Q = alpha_p*pdf_exp;
		f = bdp_bssrdf(d_r, z_r, props, C)*Q*kappa;
		s = single_diffuse(t_j, d_r, wt, normalize(xr_xo), no, props)*M_1_PIf*C.x*kappa;
		S_d += (f + s)*w_equ / (w_exp*pdf_exp + w_equ*pdf_equ);  // equiangular sampling part (t_j)
	}
	return max(S_d, 0.0) / N;
}

_fn float3 photon_beam_diffusion_bssrdf(const BSSRDFGeometry & geometry, const float recip_ior, const MaterialDataCommon& material, unsigned int flags, TEASampler & sampler)
{
    const ScatteringMaterialProperties& properties = material.scattering_properties;
	optix::float4 C = make_float4(properties.C_phi_norm, properties.C_phi, properties.C_E, properties.A);
	optix::float3 res;
	optix::float3 w12, w21;
	float R12, R21;
	bool include_fresnel_out = (flags &= BSSRDFFlags::EXCLUDE_OUTGOING_FRESNEL) == 0;
	refract(geometry.wi, geometry.ni, recip_ior, w12, R12);
	refract(geometry.wo, geometry.no, recip_ior, w21, R21);
	float F = include_fresnel_out? (1 - R21) : 1.0f;

	for (int k = 0; k < 3; k++)
	{
		optix::float4 props = optix::make_float4(optix::get_channel(k, properties.scattering), optix::get_channel(k, properties.absorption), optix::get_channel(k, properties.meancosine), 1.0f);	
		optix::get_channel(k, res) = pbd_bssrdf(geometry.xi, geometry.ni, w12, geometry.xo, geometry.no, props, C);
	}
	return res * (1 - R12) * F;
}
