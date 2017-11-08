#pragma once
#include "host_device_common.h"

#define PHOTON_STATUS_NEW 0
#define PHOTON_STATUS_SCATTERING 1

struct PhotonSample
{
	optix::float3 xp; // Position of the photon
	int i;			  // current iteration
	optix::float3 wp; // Direction of the photon
	unsigned int t;			  // current random seed
	float flux;		  // Current flux of the photon
	int status;
	optix::int2 pad;
};

__host__ __device__ __forceinline__ PhotonSample get_empty_photon()
{
	PhotonSample p;
	p.xp = optix::make_float3(0);
	p.wp = optix::make_float3(0,0,-1);
	p.i = 0;
	p.t = 0;
	p.flux = 0;
	p.status = PHOTON_STATUS_NEW;
	return p;
}
