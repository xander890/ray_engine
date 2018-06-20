#pragma once
#include "scattering_properties.h"
#include "optix_device_utils.h"

template<typename T>
_fn T erf_approx(T x) {
	const T a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
	T t = 1.0 / (1.0 + p*abs(x));
	T y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
	return copysign(1.0,x)*y;
}

template<typename T>
_fn T v0(T alpha, T sigma_s) { // Fit by Toshiya
	return 5.0*pow(0.1, -6.11*alpha*alpha + 1.168*alpha + 7.94) / (sigma_s*sigma_s);  // quantization step
}

template<typename T>
_fn T weight(T tau_i, T tau_ip1, T absorption) {
	return (exp(-tau_i*absorption) - exp(-tau_ip1*absorption)) / absorption;
}

template<typename T>
_fn T gauss2D(T v, T r_sqr) {
	return exp(-r_sqr / (2.0*v)) / (M_2PIf * v);
}

template<typename T>
_fn T weight_phi_approx(T v, T d, T alpha_p, T sigma_t) {
	// numerically more stable approximation by Eugene
	T c1 = sigma_t*(d + sigma_t*v*0.5);
	T c2 = d + sigma_t*v;
	if (c1 > 20.0) {
		T c2_sqr = c2*c2;
		T c2_p4 = c2_sqr*c2_sqr;
		return alpha_p*sigma_t / sqrt(M_2PIf)*sqrt(v)*exp(c1 - c2_sqr / (2.0*v))*(3.0*v*v - v*c2_sqr + c2_p4) / (c2_p4*c2);
	}
	return alpha_p*sigma_t*0.5*exp(c1)*(1.0 - erf_approx<T>((d + sigma_t*v) / sqrt(2.0*v)));
}

template<typename T>
_fn T weight_phi_R(T v, T alpha_p, T sigma_t, T d_e) {
	return weight_phi_approx<T>(v, 0.0, alpha_p, sigma_t) - weight_phi_approx<T>(v, d_e, alpha_p, sigma_t);
}

template<typename T>
_fn T weight_E_div_D(T v, T d, T alpha_p, T sigma_t) {
	return sigma_t*(-weight_phi_approx(v, d, alpha_p, sigma_t) + alpha_p / sqrt(M_2PIf*v)*exp(-d*d / (2.0*v)));
}

template<typename T>
_fn T weight_E_R_div_D(T v, T alpha_p, T sigma_t, T d_e) {
	return weight_E_div_D<T>(v, 0.0, alpha_p, sigma_t) + weight_E_div_D<T>(v, d_e, alpha_p, sigma_t);
}

template<typename T>
_fn T quantized_diffusion(T dist, optix::float4 props, optix::float4 C) {
	const int no_of_gaussians = 45;
	T sigma_s = props.x, sigma_a = props.y, g = props.z, A = C.w;
	T sigma_t = sigma_s + sigma_a;
	T alpha = sigma_s / sigma_t;
	T sigma_s_p = sigma_s*(1.0 - g);
	T sigma_t_p = sigma_s_p + sigma_a;
	T alpha_p = sigma_s_p / sigma_t_p;
	T D = 1.0 / (3.0*sigma_t_p);
	D *= (sigma_a*2.0 + sigma_s_p) / sigma_t_p;  // Grosjeans approximation
	T d_e = 4.0*A*D;
	T s = (1.0 + sqrt(5.0))*0.5;
	T r_sqr = dist*dist;
	T S_d = 0.0;
	T tau_i = 0.0;
	T tau_ip1 = v0(alpha, sigma_s) / D;
	for (int i = 0; i < no_of_gaussians; ++i) {
		T v = D*(tau_i + tau_ip1);
		T w_i = weight(tau_i, tau_ip1, sigma_a);
		T w_R = weight_phi_R<T>(v, alpha_p, sigma_t, d_e)*C.y + weight_E_R_div_D<T>(v, alpha_p, sigma_t, d_e)*D*C.z;
		S_d += w_R*w_i*gauss2D<T>(v, r_sqr);
		tau_i = tau_ip1;
		tau_ip1 *= s;
	}
	return M_1_PIf*alpha_p*C.x*S_d;
}

__host__ __forceinline__ void precompute_quantized_diffusion(optix::float3* dest, size_t bins, double max_dist, const ScatteringMaterialProperties & properties)
{
	optix::float4 C = optix::make_float4(properties.C_phi_norm, properties.C_phi, properties.C_E, properties.A);
	double step = max_dist / bins;

	for (size_t i = 0; i < bins; i++) {
		for (int k = 0; k < 3; k++)
		{
			optix::float4 props = optix::make_float4(optix::get_channel(k, properties.scattering), optix::get_channel(k, properties.absorption), optix::get_channel(k, properties.meancosine), 1.0f);
			optix::get_channel(k, dest[i]) = (float)quantized_diffusion<double>(step * i, props, C);
		}	
	}	
}