#pragma once
#include <optix_world.h>
#include <host_device_common.h>

#define IMPROVED_ENUM_NAME BssrdfSamplePointOnTangentTechnique
#define IMPROVED_ENUM_LIST ENUMITEM_VALUE(EXPONENTIAL_DISK,0) \
						   ENUMITEM_VALUE(NEURAL_NETWORK_IMPORTANCE_SAMPLING,1) 
#include "improved_enum.def"

#define IMPROVED_ENUM_NAME BssrdfSamplingType
#define IMPROVED_ENUM_LIST ENUMITEM_VALUE(BSSRDF_SAMPLING_CAMERA_BASED,0) \
						   ENUMITEM_VALUE(BSSRDF_SAMPLING_TANGENT_PLANE,1) \
						   ENUMITEM_VALUE(BSSRDF_SAMPLING_TANGENT_PLANE_TWO_PROBES,2) \
						   ENUMITEM_VALUE(BSSRDF_SAMPLING_MIS_AXIS,3)  
#include "improved_enum.def"

#define IMPROVED_ENUM_NAME ScatteringDipole
#define IMPROVED_ENUM_LIST	ENUMITEM_VALUE(STANDARD_DIPOLE_BSSRDF,0) \
							ENUMITEM_VALUE(DIRECTIONAL_DIPOLE_BSSRDF,1) \
							ENUMITEM_VALUE(QUANTIZED_DIFFUSION_BSSRDF,2) \
							ENUMITEM_VALUE(PHOTON_BEAM_DIFFUSION_BSSRDF,3) \
							ENUMITEM_VALUE(APPROX_STANDARD_DIPOLE_BSSRDF,4) \
							ENUMITEM_VALUE(APPROX_DIRECTIONAL_DIPOLE_BSSRDF,5) \
							ENUMITEM_VALUE(FORWARD_SCATTERING_DIPOLE_BSSRDF,6) 
#include "improved_enum.def"

#define BSSRDF_SHADERS_SHOW_ALL 0
#define BSSRDF_SHADERS_SHOW_REFRACTION 1
#define BSSRDF_SHADERS_SHOW_REFLECTION 2
#define BSSRDF_SHADERS_SHOW_COUNT 3

struct ApproximateBSSRDFProperties
{
	optix::float3 approx_property_A		DEFAULT(optix::make_float3(1));
	int pad0;
	optix::float3 approx_property_s		DEFAULT(optix::make_float3(1));
	float sampling_mfp_s;
};

struct QuantizedDiffusionProperties
{
	BufPtr1D<optix::float3> precomputed_bssrdf;
	float max_dist_bssrdf       DEFAULT(10.0f);
	int precomputed_bssrdf_size DEFAULT(1024);
	int use_precomputed_qd		DEFAULT(1);
};

struct BSSRDFSamplingProperties
{
	BssrdfSamplingType::Type        sampling_method				DEFAULT(BssrdfSamplingType::BSSRDF_SAMPLING_TANGENT_PLANE);
    BssrdfSamplePointOnTangentTechnique::Type   sampling_tangent_plane_technique    DEFAULT(BssrdfSamplePointOnTangentTechnique::EXPONENTIAL_DISK);
	int use_jacobian				DEFAULT(1);
	float d_max						DEFAULT(1.0f);
	float dot_no_ni_min				DEFAULT(0.001f);
	optix::float3 mis_weights		DEFAULT(optix::make_float3(1.f, 0.f, 0.f));
//	optix::float3 mis_weights		DEFAULT(optix::make_float3(0.5f, 0.25f, 0.25f));
	int show_mode					DEFAULT(BSSRDF_SHADERS_SHOW_ALL);
	optix::float4 mis_weights_cdf	DEFAULT(optix::make_float4(0.0f, 1.f, 1.f, 1.0f));
	//	optix::float4 mis_weights_cdf	DEFAULT(optix::make_float4(0.0f, 0.5f, 0.75f, 1.0f));
};
