#ifndef RENDERINGMETHOD_H
#define RENDERINGMETHOD_H

#include <optix_world.h>
#include <map>
#include <vector>
#include "folders.h"

class RenderingMethod 
{
public:
    virtual ~RenderingMethod() = default;
    RenderingMethod(optix::Context & context);
	virtual void init() = 0;
	virtual void pre_trace() = 0;

protected:
    optix::Context& context;
};

#endif // RENDERINGMETHOD_H
