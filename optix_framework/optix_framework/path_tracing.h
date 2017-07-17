#ifndef path_tracing_h__
#define path_tracing_h__

#pragma once
#include "rendering_method.h"


class PathTracing :
	public RenderingMethod
{
public:
	PathTracing(optix::Context & context) : RenderingMethod(context) {}

    void init() override;

    void pre_trace() override;

};



#endif // path_tracing_h__
