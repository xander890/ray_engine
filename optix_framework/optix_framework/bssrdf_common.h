#pragma once
#include <optix_world.h>
#include <host_device_common.h>

/*
	How to sample points on the disk around the normal of the outgoing point of a BSSRDF.
	Possible options include exponential or uniform disk. Otherwise a neural network sampler can 
	be used to sample points around the normal (experimental).
*/
#define IMPROVED_ENUM_NAME BssrdfSamplePointOnTangentTechnique
#define IMPROVED_ENUM_LIST ENUMITEM_VALUE(EXPONENTIAL_DISK,0) \
                           ENUMITEM_VALUE(UNIFORM_DISK,1) \
						   ENUMITEM_VALUE(NEURAL_NETWORK_IMPORTANCE_SAMPLING,2)
#include "improved_enum.inc"

/*
	How to sample points around the outgoing xo point of a scattering material.
	Options are:
		BSSRDF_SAMPLING_CAMERA_BASED	- Camera mEye based sampling (Mertens et al. [2003])
		BSSRDF_SAMPLING_TANGENT_PLANE	- Sample on a disk around the normal n_o, then project down the points.
		BSSRDF_SAMPLING_MIS_AXIS		- As tangent plane, but selects one of the three axes around the outgoing normal (normal, tangent or bitangent). Technique from King  et al. [2013].
*/
#define IMPROVED_ENUM_NAME BssrdfSamplingType
#define IMPROVED_ENUM_LIST ENUMITEM_VALUE(BSSRDF_SAMPLING_CAMERA_BASED,0) \
						   ENUMITEM_VALUE(BSSRDF_SAMPLING_TANGENT_PLANE,1) \
						   ENUMITEM_VALUE(BSSRDF_SAMPLING_MIS_AXIS,2)
#include "improved_enum.inc"

/*
Available dipoles.
STANDARD_DIPOLE_BSSRDF				- Jensen et al. [2001] standard dipole
APPROX_DIRECTIONAL_DIPOLE_BSSRDF    - Approximate directional (experimental)
DIRECTIONAL_DIPOLE_BSSRDF			- Frisvad et al. [2014] directional dipole
APPROX_STANDARD_DIPOLE_BSSRDF		- Christiensen et al. [2015] approximate dipole
QUANTIZED_DIFFUSION_BSSRDF			- d'Eon et al. [2011] quantized diffusion
PHOTON_BEAM_DIFFUSION_BSSRDF		- Habel et al. [2013] photon beam diffusion
FORWARD_SCATTERING_DIPOLE_BSSRDF	- Frederikxs et al. [2017] fully directional dipole
EMPIRICAL_BSSRDF					- Empirical in the Donner et al. [2006] format
*/
#define IMPROVED_ENUM_NAME ScatteringDipole
#define IMPROVED_ENUM_LIST	ENUMITEM_VALUE(STANDARD_DIPOLE_BSSRDF,0) \
							ENUMITEM_VALUE(DIRECTIONAL_DIPOLE_BSSRDF, 1) \
							ENUMITEM_VALUE(QUANTIZED_DIFFUSION_BSSRDF, 2) \
							ENUMITEM_VALUE(PHOTON_BEAM_DIFFUSION_BSSRDF,3) \
							ENUMITEM_VALUE(APPROX_STANDARD_DIPOLE_BSSRDF,4) \
							ENUMITEM_VALUE(APPROX_DIRECTIONAL_DIPOLE_BSSRDF,5) \
							ENUMITEM_VALUE(FORWARD_SCATTERING_DIPOLE_BSSRDF,6) \
							ENUMITEM_VALUE(EMPIRICAL_BSSRDF,7) 
#include "improved_enum.inc"

// Various flags to render BSSRDFs.
// EXCLUDE_OUTGOING_FRESNEL : Add this glass if the outgoing fresnel term must not be included. If the bssrdf does not include a fresnel term, the term is simply divided on the result. This allows importance sampling refraction saving on divisions/ catastrophic cancellations.
#define IMPROVED_ENUM_NAME BSSRDFFlags
#define IMPROVED_ENUM_LIST	ENUMITEM_VALUE(NO_FLAGS,0x00) \
							ENUMITEM_VALUE(EXCLUDE_OUTGOING_FRESNEL, 0x01) 
#include "improved_enum.inc"

#define BSSRDF_SHADERS_SHOW_ALL 0
#define BSSRDF_SHADERS_SHOW_REFRACTION 1
#define BSSRDF_SHADERS_SHOW_REFLECTION 2
#define BSSRDF_SHADERS_SHOW_COUNT 3

/*
	Properties for approximated BSSRDFs. 
	See approximate_dipoles_host.h
*/
struct ApproximateBSSRDFProperties
{
	optix::float3 approx_property_A		DEFAULT(optix::make_float3(1));
	int pad0;
	optix::float3 approx_property_s		DEFAULT(optix::make_float3(1));
	int pad1;
};

/*
 * Additional properties for the quantized diffusion model (mostly to precompute)
 * FIXME move me into some common file for quantized diffusion
 */
struct QuantizedDiffusionProperties
{
	BufPtr1D<optix::float3> precomputed_bssrdf;
	float max_dist_bssrdf       DEFAULT(10.0f);
	int precomputed_bssrdf_size DEFAULT(1024);
	int use_precomputed_qd		DEFAULT(0);
};

/*
 *  Two points on the surface with realitive attributes to evaluate the bssrdf.
 */
struct BSSRDFGeometry
{
    optix::float3 xi;
    optix::float3 wi;
    optix::float3 ni;
    optix::float3 xo;
    optix::float3 wo;
    optix::float3 no;
};

/*
 * Properties used to importance sample the BSSRDF.
 */
struct BSSRDFSamplingProperties
{
	BssrdfSamplingType::Type        			sampling_method						DEFAULT(BssrdfSamplingType::BSSRDF_SAMPLING_TANGENT_PLANE); // See above
    BssrdfSamplePointOnTangentTechnique::Type   sampling_tangent_plane_technique    DEFAULT(BssrdfSamplePointOnTangentTechnique::EXPONENTIAL_DISK); // Se above
	int 										use_jacobian						DEFAULT(0);
	float 										d_max								DEFAULT(1.0f);
	float 										dot_no_ni_min						DEFAULT(0.001f);
	int 										show_mode							DEFAULT(BSSRDF_SHADERS_SHOW_ALL);
    float                                       R_max                               DEFAULT(1.0f);
	float 										sampling_inverse_mean_free_path		DEFAULT(0.0f);
	int 										exclude_backfaces					DEFAULT(0);
};
