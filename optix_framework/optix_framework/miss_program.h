#pragma once
#include <optix_world.h>
#include "gui.h"

class MissProgram
{
public:
    MissProgram() = default;
	virtual ~MissProgram() = default;
    virtual void init(optix::Context & ctx);
    virtual void set_into_gpu(optix::Context & ctx);
    virtual void set_into_gui(GUI * gui) = 0;
    virtual void remove_from_gui(GUI * gui) = 0;
protected:
    virtual bool get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program) = 0;
private:
    bool mInit = false;
};  