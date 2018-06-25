#pragma once
#include <optix_world.h>
#include <optix.h>

/*
 * Utility class for color conversions.
 */

_fn optix::float3 Yxy2XYZ( const optix::float3& Yxy )
{
	return optix::make_float3(  Yxy.y * ( Yxy.x / Yxy.z ),
		Yxy.x,
		( 1.0f - Yxy.y - Yxy.z ) * ( Yxy.x / Yxy.z ) );
}

_fn optix::float3 XYZ2rgb( const optix::float3& xyz) 
{
	const float R = optix::dot( xyz, optix::make_float3(  3.2410f, -1.5374f, -0.4986f ) );
	const float G = optix::dot( xyz, optix::make_float3( -0.9692f,  1.8760f,  0.0416f ) );
	const float B = optix::dot( xyz, optix::make_float3(  0.0556f, -0.2040f,  1.0570f ) );
	return optix::make_float3( R, G, B );
}

_fn optix::float3 Yxy2rgb( optix::float3 Yxy )
{
	// First convert to xyz
	optix::float3 xyz = optix::make_float3( Yxy.y * ( Yxy.x / Yxy.z ), Yxy.x, ( 1.0f - Yxy.y - Yxy.z ) * ( Yxy.x / Yxy.z ) );

	const float R = optix::dot( xyz, optix::make_float3(  3.2410f, -1.5374f, -0.4986f ) );
	const float G = optix::dot( xyz, optix::make_float3( -0.9692f,  1.8760f,  0.0416f ) );
	const float B = optix::dot( xyz, optix::make_float3(  0.0556f, -0.2040f,  1.0570f ) );
	return optix::make_float3( R, G, B );
}

_fn optix::float3 rgb2Yxy( optix::float3 rgb)
{
	// convert to xyz
	const float X = optix::dot( rgb, optix::make_float3( 0.4124f, 0.3576f, 0.1805f ) );
	const float Y = optix::dot( rgb, optix::make_float3( 0.2126f, 0.7152f, 0.0722f ) );
	const float Z = optix::dot( rgb, optix::make_float3( 0.0193f, 0.1192f, 0.9505f ) );

	// convert xyz to Yxy
	return optix::make_float3( Y,
		X / ( X + Y + Z ),
		Y / ( X + Y + Z ) );
}

_fn float luminance_NTSC(optix::float3 & color)
{
    return optix::dot(color, optix::make_float3(0.2989f, 0.5866f, 0.1145f));
}

_fn optix::float3 tonemap( const optix::float3 &hdr_value, float Y_log_av, float Y_max)
{
	optix::float3 val_Yxy = rgb2Yxy( hdr_value );

	float Y        = val_Yxy.x; // Y channel is luminance
	const float a = 0.04f;
	float Y_rel = a * Y / Y_log_av;
	float mapped_Y = Y_rel * (1.0f + Y_rel / (Y_max * Y_max)) / (1.0f + Y_rel);

	optix::float3 mapped_Yxy = optix::make_float3(mapped_Y, val_Yxy.y, val_Yxy.z);
	optix::float3 mapped_rgb = Yxy2rgb( mapped_Yxy );

	return mapped_rgb;
}

_fn optix::float3 hsv2rgb(float h, float s, float v) {
	float h6 = h*6.0f;
	float frac = h6 - floor(h6);
	optix::float4 ell = v * optix::make_float4(1.0f - s, 1.0f - s*frac, 1.0f - s*(1.0f - frac), 1.0);
	return h6 < 1.0 ? optix::make_float3(ell.w, ell.z, ell.x) : (h6 < 2.0 ? optix::make_float3(ell.y, ell.w, ell.x) :
																 (h6 < 3.0 ? optix::make_float3(ell.x, ell.w, ell.z) : (h6 < 4.0 ? optix::make_float3(ell.x, ell.y, ell.w) :
																														(h6 < 5.0 ? optix::make_float3(ell.z, ell.x, ell.w) : optix::make_float3(ell.w, ell.x, ell.y)))));
}