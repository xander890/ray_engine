#pragma once
#include <optix_world.h>
#include <optix_serialize_utils.h>

class MissProgram
{
public:
    MissProgram(optix::Context & ctx) : mContext(ctx) {}
	virtual ~MissProgram() = default;
    virtual void init();
    virtual void load();
	virtual bool on_draw() = 0;

protected:
    virtual bool get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program) = 0;
	optix::Context mContext = nullptr;
	MissProgram() {};
private:
    bool mInit = false;

    friend class cereal::access;
	template<class Archive>
	void load(Archive & archive) 
	{ }
	template<class Archive>
    void save(Archive & archive) const
    { }
};

CEREAL_REGISTER_TYPE(MissProgram)