
#include <optix_world.h>
#include "light_host.h"

void SingularLight::init(optix::Context &context)
{
    mContext = context;
}

void SingularLight::set_into_gpu()
{

}

bool SingularLight::on_draw()
{
    return false;
}

