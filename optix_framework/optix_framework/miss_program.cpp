#include "miss_program.h"
#include "host_device_common.h"

void MissProgram::init(optix::Context & ctx)
{
    optix::Program program;
    for (int i = 0; i < RAY_TYPE_COUNT; i++)
    {
        if (get_miss_program(i, ctx, program))
        {
            ctx->setMissProgram(i, program);
        }
    }
    mInit = true;
}

void MissProgram::set_into_gpu(optix::Context & ctx)
{
    if (!mInit)
        init(ctx);
}