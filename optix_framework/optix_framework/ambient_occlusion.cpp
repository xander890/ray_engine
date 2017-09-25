#include "ambient_occlusion.h"
#include "parameter_parser.h"
using namespace optix;

void AmbientOcclusion::init()
{
	unsigned int N = (unsigned int)ConfigParameters::get_parameter<int>("config","N", 1, "Monte carlo samples.");
	context["N"]->setUint(N);
}

void AmbientOcclusion::pre_trace()
{
	return;
}
