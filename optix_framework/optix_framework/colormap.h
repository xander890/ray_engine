#pragma once
/*
Return a RGB colour value given a scalar v in the range [vmin,vmax]
In this case each colour component ranges from 0 (no contribution) to
1 (fully saturated), modifications for other ranges is trivial.
The colour is clipped at the end of the scales if v is outside
the range [vmin,vmax]
*/
#include <optix_world.h>
#include "host_device_common.h"

_fn optix::float3 jet(float v, float vmin = 0.0f, float vmax = 1.0f)
{
	optix::float3 c = optix::make_float3(1.0,1.0,1.0); // white
	float dv;

	if (v < vmin)
		v = vmin;
	if (v > vmax)
		v = vmax;
	dv = vmax - vmin;

	if (v < (vmin + 0.25 * dv)) {
		c.x = 0;
		c.y = 4 * (v - vmin) / dv;
	}
	else if (v < (vmin + 0.5 * dv)) {
		c.x = 0;
		c.z = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
	}
	else if (v < (vmin + 0.75 * dv)) {
		c.x = 4 * (v - vmin - 0.5 * dv) / dv;
		c.z = 0;
	}
	else {
		c.y = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
		c.z = 0;
	}

	return(c);
}

_fn optix::float3 hsv2rgb(float h, float s, float v) {
  float h6 = h*6.0;
  float frac = h6 - floor(h6);
  optix::float4 ell = v * optix::make_float4(1.0 - s, 1.0 - s*frac, 1.0 - s*(1.0 - frac), 1.0);
  return h6 < 1.0 ? optix::make_float3(ell.w, ell.z, ell.x) : (h6 < 2.0 ? optix::make_float3(ell.y, ell.w, ell.x) : 
	  (h6 < 3.0 ? optix::make_float3(ell.x, ell.w, ell.z) : (h6 < 4.0 ? optix::make_float3(ell.x, ell.y, ell.w) : 
	  (h6 < 5.0 ? optix::make_float3(ell.z, ell.x, ell.w) : optix::make_float3(ell.w, ell.x, ell.y)))));
}