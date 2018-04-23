#ifndef SIMPLETRACING_H
#define SIMPLETRACING_H

#include "rendering_method.h"

class SimpleTracing : public RenderingMethod 
{
public:
    SimpleTracing(optix::Context & context) : RenderingMethod(context) {}
	virtual void init();
	std::string get_suffix() const { return ""; }
};



#endif // SIMPLETRACING_H
