#pragma once
#include "miss_program.h"

class ConstantBackground : public MissProgram
{
public:
    ConstantBackground(const optix::float3& bg) : background_color(bg) {}
    virtual ~ConstantBackground() {}

    virtual void init(optix::Context & ctx) override;
    virtual void set_into_gpu(optix::Context & ctx) override;
	virtual void on_draw() override;
private:
    virtual bool get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program) override;
    optix::float3 background_color;
};

