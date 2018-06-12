#pragma once
#include "host_device_common.h"
#include "random.h"
#include "empirical_bssrdf_common.h"
#define PHOTON_STATUS_NEW 0
#define PHOTON_STATUS_SCATTERING 1

//#define INCLUDE_SINGLE_SCATTERING/
//#define INCLUDE_GEOMETRIC_TERM
#define ACCURATE_RANDOM
#define TERMINATE_ON_SMALL_FLUX

#ifdef ACCURATE_RANDOM
#define RND_FUNC rnd_accurate
#define SEED_TYPE Seed64
#define PAD_STRUCT
__device__ __forceinline__ void init_seed(SEED_TYPE & seed, unsigned long long q) { seed.l = q; }
#else
#define RND_FUNC rnd_tea
#define SEED_TYPE optix::uint
#define PAD_STRUCT optix::uint pad;
__device__ __forceinline__ void init_seed(SEED_TYPE & seed, unsigned long long q) { seed = (unsigned int)q; }
#endif

#define IMPROVED_ENUM_NAME IntegrationMethod
#define IMPROVED_ENUM_LIST ENUMITEM_VALUE(MCML,0) ENUMITEM_VALUE(CONNECTIONS,1) ENUMITEM_VALUE(CONNECTIONS_WITH_FIX,2)
#include "improved_enum.def"

#define IMPROVED_ENUM_NAME BiasMode
#define IMPROVED_ENUM_LIST ENUMITEM_VALUE(BIAS_ONLY,0) ENUMITEM_VALUE(BIASED_RESULT,1) ENUMITEM_VALUE(RENDER_ALL,2)
#include "improved_enum.def"

struct PhotonSample
{
	optix::float3 xp; // Position of the photon
	int i;			  // current iteration
	optix::float3 wp; // Direction of the photon
	float flux;		  // Current flux of the photon
    SEED_TYPE t;			  // current random seed
	int status;
    float G;
    PAD_STRUCT
};

_fn PhotonSample get_empty_photon()
{
	PhotonSample p;
	p.xp = optix::make_float3(0);
	p.wp = optix::make_float3(0,0,-1);
	p.i = 0;
	p.flux = 0;
	p.status = PHOTON_STATUS_NEW;
    p.G = 1;
	return p;
}

struct BSSRDFRendererData
{
    float mThetai       DEFAULT(0.0f);
    optix::float2 mThetas       DEFAULT(optix::make_float2(0.0f, 7.5));
    optix::float2 mRadius       DEFAULT(optix::make_float2(0.0f, 1.f));
    float mArea;
    rtBufferId<float, 2> mWeightedSolidAngleBuffer;
	float mDeltaR;
	float mDeltaThetas;
};

struct BSSRDFSimulatedOptions
{
    float mBias                                     DEFAULT(-1.0f);
    BiasMode::Type  mbBiasMode                      DEFAULT(BiasMode::RENDER_ALL);
    IntegrationMethod::Type mIntegrationMethod      DEFAULT(IntegrationMethod::MCML);
	bool mbCosineWeighted							DEFAULT(false);
};