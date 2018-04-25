#pragma once
#include "miss_program.h"
#include "optix_serialize.h"

class ConstantBackground : public MissProgram
{
public:
    ConstantBackground(const optix::float3& bg = optix::make_float3(0.5f)) : background_color(bg) {}
    virtual ~ConstantBackground() {}

    virtual void init(optix::Context & ctx) override;
    virtual void set_into_gpu(optix::Context & ctx) override;
	virtual bool on_draw() override;
private:
    virtual bool get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program) override;
    optix::float3 background_color;

    friend class cereal::access;
    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(cereal::virtual_base_class<MissProgram>(this), CEREAL_NVP(background_color));
    }
};


CEREAL_REGISTER_TYPE(ConstantBackground)