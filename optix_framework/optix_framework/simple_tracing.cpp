#include "simple_tracing.h"
#include "material_library.h"
#include "optics_utils.h"


void SimpleTracing::init(optix::Context &ctx)
{
    RenderingMethod::init(ctx);
}
