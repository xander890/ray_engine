#ifndef RANDOM_H
#define RANDOM_H
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optix_world.h>
#include "math_helpers.h"
#include "md5.h"

 // TEA algorithm https://www.csee.umbc.edu/~olano/class/635-11-2/lsebald1.pdf
template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
{
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for( unsigned int n = 0; n < N; n++ )
  {
    s0 += 0x9e3779b9;
    v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
    v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
  }

  return v0;
}

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}

// Hash hack http://www.ci.i.u-tokyo.ac.jp/~hachisuka/tdf2015.pdf
static __host__ __device__ __inline__ float hash_tdf(const optix::float3 idx, float grid_scale, int hash_num)
{
	// use the same procedure as GPURnd
	optix::float4 n = optix::make_float4(idx, grid_scale * 0.5f) * 4194304.0f / grid_scale;

	const optix::float4 q = optix::make_float4(1225.0f, 1585.0f, 2457.0f, 2098.0f);
	const optix::float4 r = optix::make_float4(1112.0f, 367.0f, 92.0f, 265.0f);
	const optix::float4 a = optix::make_float4(3423.0f, 2646.0f, 1707.0f, 1999.0f);
	const optix::float4 m = optix::make_float4(4194287.0f, 4194277.0f, 4194191.0f, 4194167.0f);

	optix::float4 beta = floor(n / q);
	optix::float4 p = a * (n - beta * q) - beta * r;
	beta = (signf(-p) + optix::make_float4(1.0f)) * optix::make_float4(0.5f) * m;
	n = (p + beta);

	return floor(fracf(dot(n / m, optix::make_float4(1.0f, -1.0f, 1.0f, -1.0f))) * hash_num);
}


static __host__ __device__ __inline__ unsigned int hash(unsigned int &prev)
{
	optix::uint4 md5 = rand_md5(prev, 0);
	prev = md5.x;
	return prev & 0x7FFFFFFF;
}

static __host__ __device__ __inline__ float rnd_accurate(unsigned int &prev)
{
	return ((float)hash(prev) / (float)0x80000000);
}

static __host__ __device__ __inline__ unsigned int tea_hash(unsigned int &prev)
{
	prev = tea<16>(prev, 100);
	return prev & 0x7FFFFFFF;
}

static __host__ __device__ __inline__ float rnd_tea(unsigned int &prev)
{
	return ((float)tea_hash(prev) / (float)0x80000000);
}


#endif