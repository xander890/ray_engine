#pragma once
#include "folders.h"
#include "optix_serialize.h"
#include <optix_world.h>

/*
Class abstraction to define a method to render the scene. 
Example method supported in this framework are recursive ray tracing or path tracing, tough more can be supported. 

Each method is identified by a suffix that is used to look for the method in the .cu file When loading the program. As example, path tracing has suffix "_path_tracing", so the following method will be loaded in the closest hit program:

RT_PROGRAM void shade_path_tracing()
{
}

If a method is created, it has a init method, a pre_trace and a post_trace method. Use these to perform any preprocessing related to the method

*/
class RenderingMethod 
{
public:
    virtual ~RenderingMethod() = default;
    RenderingMethod() {}
	virtual std::string get_suffix() const = 0; // Redeclare this method in derived classes.

	virtual void init(optix::Context & ctx) { context = ctx; }
	virtual void pre_trace() {}
	virtual void post_trace() {}

protected:
    optix::Context context;

	// No particular serialization needed here. 
	// Redeclare this method in deriving classes to load relevant data.
    template<class Archive>
    void serialize(Archive & archive)
    {
    }
};



