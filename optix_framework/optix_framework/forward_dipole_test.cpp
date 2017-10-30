#pragma warning(disable : 4244)
#include "forward_dipole_test.h"
#include "scattering_properties.h"
#include "optix_helpers.h"
#include "forward_dipole.h"


void test_forward_dipole()
{
	test_forward_dipole_cuda();
}