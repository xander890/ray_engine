#ifndef SIMPLETRACING_H
#define SIMPLETRACING_H

#include "rendering_method.h"

class SimpleTracing : public RenderingMethod 
{
public:
    SimpleTracing(optix::Context & context) : RenderingMethod(context) {}
	virtual void init();
	virtual void pre_trace();
};



#endif // SIMPLETRACING_H
