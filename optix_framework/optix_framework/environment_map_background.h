#pragma once
#include "miss_program.h"
#include "environment_map.h"
#include "immediate_gui.h"

class EnvironmentMap : public MissProgram
{
public:
    EnvironmentMap(std::string envmap_file) : envmap_file(envmap_file), texture_width(0), texture_height(0),
                                              camera_1(0), camera_2(0), camera_3(0)
	{
	}

    virtual ~EnvironmentMap() {}

    virtual void init(optix::Context & ctx) override;
    virtual void set_into_gpu(optix::Context & ctx) override;
    virtual bool on_draw() override;
private:
    virtual bool get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program) override;
    
    EnvmapProperties properties;
    EnvmapImportanceSamplingData sampling_properties;

    optix::float3 envmap_deltas;
    optix::TextureSampler environment_sampler;
    optix::Context context;
    optix::Buffer property_buffer;
    optix::Buffer sampling_property_buffer;
    std::string envmap_file;
    int texture_width, texture_height;
    int camera_1, camera_2, camera_3;
	
    void presample_environment_map();
    bool resample_envmaps = true;

};

