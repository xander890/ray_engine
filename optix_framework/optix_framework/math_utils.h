
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
#include "host_device_common.h"

// Constants
#ifndef M_4PIf
#define M_4PIf		12.5663706143591729538f
#endif
#ifndef M_1_4PIf
#define M_1_4PIf	0.07957747154594766788f
#endif
#ifndef M_1_4PIPIf
#define M_1_4PIPIf	0.02533029591058444286f
#endif
#ifndef M_2PIf
#define M_2PIf		6.28318530717958647692f
#endif


_fn float normalize_angle(float deg)
{
	return deg - 2 * M_PIf * floor(deg / (2 * M_PIf));
}

_fn float deg2rad(float deg)
{
	return deg * M_PIf / 180.0f;
}

_fn float rad2deg(float rad)
{
	return rad * 180.0f / M_PIf;
}

_fn float signf(float v)
{
	return copysignf(1.0f, v);
}

_fn float fracf(float x)
{
	return x - truncf(x);
}

_fn void rotate_to_normal(const optix::float3& normal, optix::float3& v)
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

_fn optix::float3 spherical_to_cartesian(const float theta_o, const float phi_o)
{
	return optix::make_float3(cosf(phi_o)*sinf(theta_o), sinf(phi_o)*sinf(theta_o), cosf(theta_o));
}
 
_fn optix::float2 cartesian_to_spherical(const optix::float3& v)
{
    return optix::make_float2(acosf(v.z), atan2f(v.y, v.x));
}

_fn optix::float2 direction_to_uv_coord_cubemap(const optix::float3& direction, const optix::Matrix3x3& rotation = optix::Matrix3x3::identity())
{
	optix::float3 dir = rotation * direction;
	return optix::make_float2(0.5f + 0.5f * (atan2f(dir.x, -dir.z) * M_1_PIf), acosf(-dir.y) * M_1_PIf);
}

_fn void create_onb(const optix::float3& n, optix::float3& b1, optix::float3& b2)
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

_fn void create_onb(const optix::float3& n, optix::float3& U, optix::float3& V, optix::float3& W)
{
  W = optix::normalize(n);
  create_onb(W, U, V);
}

_fn optix::float3 rotate_around(const optix::float3& vector, const optix::float3& axis, const float angle)
{
	// Rodrigues formula
	const float cos_theta = cosf(angle);
	return cos_theta * vector + optix::cross(axis, vector) * sinf(angle) + axis * optix::dot(axis, vector) * (1 - cos_theta);
}

/*
// Create ONB from normal.  Resulting W is parallel to normal
static
_fn void create_onb( const optix::float3& n, optix::float3& U, optix::float3& V, optix::float3& W )
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
_fn void create_onb( const optix::float3& n, optix::float3& U, optix::float3& V)
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
_fn optix::float3 project_on_plane(const optix::float3& n, const optix::float3& v)
{
	return optix::normalize(v - n * optix::dot(n, v));
}

#define GENERATE_VECTORIZED(fun) \
_fn optix::float2 fun(const optix::float2 & v) { return optix::make_float2(fun(v.x), fun(v.y)); } \
_fn optix::float3 fun(const optix::float3 & v) { return optix::make_float3(fun(v.x), fun(v.y), fun(v.z)); } \
_fn optix::float4 fun(const optix::float4 & v) { return optix::make_float4(fun(v.x), fun(v.y), fun(v.z), fun(v.w)); }

GENERATE_VECTORIZED(exp)
GENERATE_VECTORIZED(sqrt)
GENERATE_VECTORIZED(abs)
GENERATE_VECTORIZED(deg2rad)
GENERATE_VECTORIZED(rad2deg)
GENERATE_VECTORIZED(signf)
GENERATE_VECTORIZED(fracf)

_fn optix::float4 max(const optix::float4 &value1, const optix::float4 &value2)
{
	return optix::make_float4(optix::fmaxf(value1.x, value2.x), optix::fmaxf(value1.y, value2.y), optix::fmaxf(value1.z, value2.z), optix::fmaxf(value1.w, value2.w));
}

_fn optix::float3 max(const optix::float3 &value1, const optix::float3 &value2)
{
	  return optix::make_float3(optix::fmaxf(value1.x,value2.x),optix::fmaxf(value1.y,value2.y),optix::fmaxf(value1.z,value2.z));
}

_fn optix::float4 min(const optix::float4 &value1, const optix::float4 &value2)
{
	return optix::make_float4(optix::fminf(value1.x, value2.x), optix::fminf(value1.y, value2.y), optix::fminf(value1.z, value2.z), optix::fminf(value1.w, value2.w));
}

_fn optix::float3 min(const optix::float3 &value1, const optix::float3 &value2)
{
	  return optix::make_float3(optix::fminf(value1.x,value2.x),optix::fminf(value1.y,value2.y),optix::fminf(value1.z,value2.z));
}


_fn optix::float3 pow(const optix::float3 &value1, const optix::float3 &exp)
{
	return optix::make_float3(powf(value1.x, exp.x), powf(value1.y, exp.y), powf(value1.z, exp.z));
}
_fn optix::float3 pow(const optix::float3 &value1, const float exp)
{
	return optix::make_float3(powf(value1.x, exp), powf(value1.y, exp), powf(value1.z, exp));
}


_fn float step(const float &edge, const float &x)
{
	  return (x < edge)? 0.0f : 1.0f;
}

_fn void sincosf(float x, float& sin_x, float& cos_x)
{
	sin_x = sinf(x);
	cos_x = cosf(x);
}
