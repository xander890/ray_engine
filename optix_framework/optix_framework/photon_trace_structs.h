#pragma once
#include "host_device_common.h"

struct PhotonSample
{
	optix::float3 xp; // Position of the photon
	int i;			  // current iteration
	optix::float3 wp; // Direction of the photon
	unsigned int t;			  // current random seed
	float flux;		  // Current flux of the photon
};
