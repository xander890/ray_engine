#include "simple_tracing.h"
#include "material_library.h"
#include "optical_helper.h"


void SimpleTracing::init(optix::Context &ctx)
{
    RenderingMethod::init(ctx);
}
