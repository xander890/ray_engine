#include "ambient_occlusion.h"

void AmbientOcclusion::pre_trace()
{
	return;
}

void AmbientOcclusion::init(optix::Context &context)
{
    RenderingMethod::init(context);
}
