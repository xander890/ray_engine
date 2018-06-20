#include "miss_program.h"
#include "host_device_common.h"

void MissProgram::init()
{
    optix::Program program;
    for (int i = 0; i < RayType::count(); i++)
    {
        if (get_miss_program(i, mContext, program))
        {
			mContext->setMissProgram(i, program);
        }
    }
    mInit = true;
}

void MissProgram::load()
{
    if (!mInit)
        init();
}