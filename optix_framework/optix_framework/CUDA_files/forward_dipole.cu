#include  "host_device_common.h"
#include <device_common_data.h>
#include <optix_helpers.h>
#include  "optix_device.h"
#include "forward_dipole.h"

RT_CALLABLE_PROGRAM Float sampleLengthDipoleProgram(
	const ForwardDipoleMaterial material,
	const ForwardDipoleProperties props,
	Float3 uL, Float3 nL, Float3 R,
	const Float3 *u0, Float3 n0,
	Float &s, TEASampler * sampler)
{
	return sampleLengthDipole(material, props, uL, nL, R, u0, n0, s, sampler);
}

RT_CALLABLE_PROGRAM Float evalMonopoleProgram(
	const ForwardDipoleMaterial material,
	Float3 u0, Float3 uL, Float3 R, Float length)
{
	return evalMonopole(material, u0, uL, R, length);
}

RT_CALLABLE_PROGRAM Float evalDipoleProgram(
	const ForwardDipoleMaterial material,
	const ForwardDipoleProperties props,
	Float3 n0, Float3 u0, 
	Float3 nL, Float3 uL,
	Float3 R, Float length
)
{
	return evalDipole(material, props, n0, u0, nL, uL, R, length);
}

__global__ void stub()
{
	ForwardDipoleMaterial mat;
	ForwardDipoleProperties props;
	Float3 t;
	Float s; unsigned int tt;
	evalMonopoleProgram(mat, t, t, t, 0);
	sampleLengthDipoleProgram(mat, props, t, t, t, &t, t, s, nullptr);
	evalDipoleProgram(mat, props, t, t, t, t, t, 0);
}