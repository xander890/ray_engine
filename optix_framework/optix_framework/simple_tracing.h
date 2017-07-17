#ifndef SIMPLETRACING_H
#define SIMPLETRACING_H

#include <optix_world.h>
#include "rendering_method.h"

using namespace optix;

class SimpleTracing : public RenderingMethod 
{
public:
	SimpleTracing(Context & context) : RenderingMethod(context) {}
	virtual void init();
	virtual void pre_trace();
};



#endif // SIMPLETRACING_H
