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
