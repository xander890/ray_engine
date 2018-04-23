#pragma once
#include "rendering_method.h"

class AmbientOcclusion :
	public RenderingMethod
{
public:
	AmbientOcclusion(optix::Context & context) : RenderingMethod(context) {}

	virtual void init();

	virtual void pre_trace();
	std::string get_suffix() const { return "_ao"; }

};

