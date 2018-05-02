#include "ambient_occlusion.h"
#include "parameter_parser.h"
using namespace optix;


void AmbientOcclusion::pre_trace()
{
	return;
}

void AmbientOcclusion::init(optix::Context &context)
{
    RenderingMethod::init(context);
}
