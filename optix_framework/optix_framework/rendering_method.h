#pragma once
#include <optix_world.h>
#include <map>
#include <vector>
#include "folders.h"
#include "optix_serialize.h"

class RenderingMethod 
{
public:
    virtual ~RenderingMethod() = default;
    RenderingMethod() {}
	virtual void init(optix::Context & ctx) { context = ctx; }
	virtual std::string get_suffix() const=0;
	virtual void pre_trace() {}
	virtual void post_trace() {}

protected:
    optix::Context context;

    template<class Archive>
    void serialize(Archive & archive)
    {
    }
};



