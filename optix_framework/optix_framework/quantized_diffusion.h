#pragma once
#include <device_common_data.h>
#include <scattering_properties.h>
#include <optix_helpers.h>

using optix::float3;

#ifndef M_4PIf
#define M_4PIf 4.0*M_PIf
#endif
#ifndef M_1_4PIf
#define M_1_4PIf 1.0 / M_4PIf
#endif
#ifndef M_1_4PIPIf
#define M_1_4PIPIf M_1_4PIf*M_1_PIf
#endif
#ifndef M_2PIf
#define M_2PIf 2.0*M_PIf
#endif

__device__ __forceinline__ float erf_approx(float x) {
	const float a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
	float t = 1.0 / (1.0 + p*abs(x));
	float y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
	return signf(x)*y;
}

__device__ __forceinline__ float v0(float alpha, float sigma_s) { // Fit by Toshiya
	return 5.0*pow(0.1, -6.11*alpha*alpha + 1.168*alpha + 7.94) / (sigma_s*sigma_s);  // quantization step
}

__device__ __forceinline__ float weight(float tau_i, float tau_ip1, float absorption) {
	return (exp(-tau_i*absorption) - exp(-tau_ip1*absorption)) / absorption;
}

__device__ __forceinline__ float gauss2D(float v, float r_sqr) {
	return expf(-r_sqr / (2.0*v)) / (M_2PIf * v);
}

__device__ __forceinline__ float weight_phi_approx(float v, float d, float alpha_p, float sigma_t) {
	// numerically more stable approximation by Eugene
	float c1 = sigma_t*(d + sigma_t*v*0.5);
	float c2 = d + sigma_t*v;
	if (c1 > 20.0) {
		float c2_sqr = c2*c2;
		float c2_p4 = c2_sqr*c2_sqr;
		return alpha_p*sigma_t / sqrtf(M_2PIf)*sqrt(v)*exp(c1 - c2_sqr / (2.0*v))*(3.0*v*v - v*c2_sqr + c2_p4) / (c2_p4*c2);
	}
	return alpha_p*sigma_t*0.5*exp(c1)*(1.0 - erf_approx((d + sigma_t*v) / sqrt(2.0*v)));
}

__device__ __forceinline__ float weight_phi_R(float v, float alpha_p, float sigma_t, float d_e) {
	return weight_phi_approx(v, 0.0, alpha_p, sigma_t) - weight_phi_approx(v, d_e, alpha_p, sigma_t);
}

__device__ __forceinline__ float weight_E_div_D(float v, float d, float alpha_p, float sigma_t) {
	return sigma_t*(-weight_phi_approx(v, d, alpha_p, sigma_t) + alpha_p / sqrtf(M_2PIf*v)*exp(-d*d / (2.0*v)));
}

__device__ __forceinline__ float weight_E_R_div_D(float v, float alpha_p, float sigma_t, float d_e) {
	return weight_E_div_D(v, 0.0, alpha_p, sigma_t) + weight_E_div_D(v, d_e, alpha_p, sigma_t);
}

__device__ __forceinline__ float quantized_diffusion(float dist, optix::float4 props, optix::float4 C) {
	const int no_of_gaussians = 45;
	float sigma_s = props.x, sigma_a = props.y, g = props.z, A = C.w;
	float sigma_t = sigma_s + sigma_a;
	float alpha = sigma_s / sigma_t;
	float sigma_s_p = sigma_s*(1.0 - g);
	float sigma_t_p = sigma_s_p + sigma_a;
	float alpha_p = sigma_s_p / sigma_t_p;
	float D = 1.0 / (3.0*sigma_t_p);
	D *= (sigma_a*2.0 + sigma_s_p) / sigma_t_p;  // Grosjeans approximation
	float d_e = 4.0*A*D;
	float s = (1.0 + sqrt(5.0))*0.5;
	float r_sqr = dist*dist;
	float S_d = 0.0;
	float tau_i = 0.0;
	float tau_ip1 = v0(alpha, sigma_s) / D;
	for (int i = 0; i < no_of_gaussians; ++i) {
		float v = D*(tau_i + tau_ip1);
		float w_i = weight(tau_i, tau_ip1, sigma_a);
		float w_R = weight_phi_R(v, alpha_p, sigma_t, d_e)*C.y + weight_E_R_div_D(v, alpha_p, sigma_t, d_e)*D*C.z;
		S_d += w_R*w_i*gauss2D(v, r_sqr);
		tau_i = tau_ip1;
		tau_ip1 *= s;
	}
	return M_1_PIf*alpha_p*C.x*S_d;
}



__device__ __forceinline__ float3 quantized_diffusion_bssrdf(float dist, const ScatteringMaterialProperties& properties)
{
	optix::float4 C = make_float4(properties.C_phi_norm, properties.C_phi, properties.C_E, properties.A);

	optix::float3 res;
	for (int k = 0; k < 3; k++)
	{
		optix::float4 props = optix::make_float4(optix::get_channel(k, properties.scattering), optix::get_channel(k, properties.absorption), optix::get_channel(k, properties.meancosine), 1.0f);
		optix::get_channel(k, res) = quantized_diffusion(dist, props, C);
	}
	return res;

}