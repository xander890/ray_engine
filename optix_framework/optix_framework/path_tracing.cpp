#include "path_tracing.h"
#include "parameter_parser.h"

void PathTracing::pre_trace()
{
}

void PathTracing::init(optix::Context & ctx)
{
	RenderingMethod::init(ctx);
	N = 1;
	context["N"]->setUint(N);
}
