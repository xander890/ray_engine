#ifndef RENDERINGMETHOD_H
#define RENDERINGMETHOD_H

#include <optix_world.h>
#include <map>
#include <vector>
#include "folders.h"
#include "mesh.h"

using namespace optix;
using namespace std;

class RenderingMethod 
{
public:
    virtual ~RenderingMethod() = default;
    RenderingMethod(Context & context);
	virtual void init() = 0;
	virtual void pre_trace() = 0;

protected:
	Context& context;
};

#endif // RENDERINGMETHOD_H
