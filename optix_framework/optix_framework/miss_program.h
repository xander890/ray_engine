#pragma once
#include <optix_world.h>


class MissProgram
{
public:
    MissProgram() = default;
	virtual ~MissProgram() = default;
    virtual void init(optix::Context & ctx);
    virtual void set_into_gpu(optix::Context & ctx);
	virtual void on_draw() = 0;
protected:
    virtual bool get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program) = 0;
private:
    bool mInit = false;
};  