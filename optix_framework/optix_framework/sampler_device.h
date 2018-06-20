#pragma once
#include "host_device_common.h"
#include "random_device.h"

class TEASampler
{
public:
	__device__ __host__ TEASampler(unsigned int seed1, unsigned int seed2)
	{
		seed = tea<16>(seed1, seed2);
	}

	__device__ __host__ TEASampler(optix::uint2 & seed) : TEASampler(seed.x, seed.y) {}

	__device__ __host__ float next1D()
	{
		return rnd_tea(seed);
	}

	__device__ __host__ optix::float2 next2D()
	{
		return optix::make_float2(rnd_tea(seed), rnd_tea(seed));
	}

	unsigned int seed;
};