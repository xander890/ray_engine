
/*
 * Copyright (c) 2008 - 2010 NVIDIA Corporation.  All rights reserved.
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

#pragma once

#include <optix_world.h>
#include <optix.h>

// Convert a float3 in [0,1)^3 to a uchar4 in [0,255]^4 -- 4th channel is set to 255
#ifdef __CUDACC__
static __device__ __inline__ optix::uchar4 make_color(const optix::float3& c)
{
    return optix::make_uchar4( static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
                               static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
                               static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
                               255u);                                                 /* A */
}
#endif

__host__ __device__ __inline__ float normalize_angle(float deg)
{
	return deg - 2 * M_PIf * std::floor(deg / (2 * M_PIf));
}

__host__ __device__ __inline__ float deg2rad(float deg)
{
	return deg * M_PIf / 180.0f;
}

__host__ __device__ __inline__ float rad2deg(float rad)
{
	return rad * 180.0f / M_PIf;
}

__host__ __device__ __inline__ void rotate_to_normal(const optix::float3& normal, optix::float3& v)
{
	if(normal.z < -0.999999f)
	{
		v = optix::make_float3(-v.y, -v.x, -v.z);
		return;
	}
	const float a = 1.0f/(1.0f + normal.z);
	const float b = -normal.x*normal.y*a;
	v = optix::make_float3(1.0f - normal.x*normal.x*a, b, -normal.x)*v.x 
	+ optix::make_float3(b, 1.0f - normal.y*normal.y*a, -normal.y)*v.y 
	+ normal*v.z;
}



__host__ __device__ __inline__ optix::float2 direction_to_uv_coord_cubemap(const optix::float3& direction, const optix::Matrix3x3& rotation = optix::Matrix3x3::identity())
{
	optix::float3 dir = rotation * direction;
	return optix::make_float2(0.5f + 0.5f * (atan2f(dir.x, -dir.z) * M_1_PIf), acosf(-dir.y) * M_1_PIf);
}

static __host__ __device__ __inline__ void create_onb(const optix::float3& n, optix::float3& b1, optix::float3& b2)
{
  if(n.z < -0.999999f) // Handle the singularity
  {
    b1 = optix::make_float3(0.0f, -1.0f, 0.0f);
    b2 = optix::make_float3(-1.0f, 0.0f, 0.0f);
    return;
  }
  const float a = 1.0f /(1.0f + n.z);
  const float b = -n.x*n.y*a;
  b1 = optix::make_float3(1.0f - n.x*n.x*a, b, -n.x);
  b2 = optix::make_float3(b, 1.0f - n.y*n.y*a, -n.y);
}

static __host__ __device__ __inline__ void create_onb(const optix::float3& n, optix::float3& U, optix::float3& V, optix::float3& W)
{
  W = optix::normalize(n);
  create_onb(W, U, V);
}

/*
// Create ONB from normal.  Resulting W is parallel to normal
static
__host__ __device__ __inline__ void create_onb( const optix::float3& n, optix::float3& U, optix::float3& V, optix::float3& W )
{
    W = normalize( n );
  U = cross( W, optix::make_float3( 0.0f, 1.0f, 0.0f ) );

  if ( fabs( U.x ) < 0.001f && fabs( U.y ) < 0.001f && fabs( U.z ) < 0.001f  )
    U = cross( W, optix::make_float3( 1.0f, 0.0f, 0.0f ) );

  U = normalize( U );
  V = cross( W, U );
}

// Create ONB from normalized vector
static
__device__ __inline__ void create_onb( const optix::float3& n, optix::float3& U, optix::float3& V)
{
  
  U = cross( n, make_float3( 0.0f, 1.0f, 0.0f ) );

  if ( dot( U, U ) < 1e-3f )
    U = cross( n, make_float3( 1.0f, 0.0f, 0.0f ) );

  U = normalize( U );
  V = cross( n, U );
}
*/
// Projects and normalizes vector on plane defined by normal
static
__device__ __inline__ optix::float3 project_on_plane(const optix::float3& n, const optix::float3& v)
{
	return optix::normalize(v - n * optix::dot(n, v));
}

static __host__ __device__ __inline__ optix::float4 max(const optix::float4 &value1, const optix::float4 &value2)
{
	return optix::make_float4(optix::fmaxf(value1.x, value2.x), optix::fmaxf(value1.y, value2.y), optix::fmaxf(value1.z, value2.z), optix::fmaxf(value1.w, value2.w));
}

static __host__ __device__ __inline__ optix::float3 max(const optix::float3 &value1, const optix::float3 &value2)
{
	  return optix::make_float3(optix::fmaxf(value1.x,value2.x),optix::fmaxf(value1.y,value2.y),optix::fmaxf(value1.z,value2.z));
}


static __host__ __device__ __inline__ optix::float4 min(const optix::float4 &value1, const optix::float4 &value2)
{
	return optix::make_float4(optix::fminf(value1.x, value2.x), optix::fminf(value1.y, value2.y), optix::fminf(value1.z, value2.z), optix::fminf(value1.w, value2.w));
}

static __host__ __device__ __inline__ optix::float3 min(const optix::float3 &value1, const optix::float3 &value2)
{
	  return optix::make_float3(optix::fminf(value1.x,value2.x),optix::fminf(value1.y,value2.y),optix::fminf(value1.z,value2.z));
}

static __host__ __device__ __inline__ optix::float3 exp(const optix::float3 &value1)
{
	  return optix::make_float3(exp(value1.x),exp(value1.y),exp(value1.z));
}

static __host__ __device__ __inline__ optix::float3 sqrt(const optix::float3 &value1)
{
	  return optix::make_float3(sqrt(value1.x),sqrt(value1.y),sqrt(value1.z));
}

static __host__ __device__ __inline__ optix::float3 abs(const optix::float3 &value1)
{
	  return optix::make_float3(abs(value1.x),abs(value1.y),abs(value1.z));
}

static __host__ __device__ __inline__ optix::float3 pow(const optix::float3 &value1, const optix::float3 &exp)
{
	return optix::make_float3(powf(value1.x, exp.x), powf(value1.y, exp.y), powf(value1.z, exp.z));
}
static __host__ __device__ __inline__ optix::float3 pow(const optix::float3 &value1, const float exp)
{
	return optix::make_float3(powf(value1.x, exp), powf(value1.y, exp), powf(value1.z, exp));
}

static __host__ __device__ __inline__ bool isnan(const optix::float3 &value1)
{
	return isnan(value1.x) || isnan(value1.y) || isnan(value1.z);
}


static __host__ __device__ __inline__ float step(const float &edge, const float &x)
{
	  return (x < edge)? 0.0f : 1.0f;
}

static __host__ __device__ __inline__ float DtoR(float d)
{
	return d*(M_PIf / 180.f);
}


static __host__ __device__ __inline__ float RtoD(float r)
{
	return r*(180.f / M_PIf);
}