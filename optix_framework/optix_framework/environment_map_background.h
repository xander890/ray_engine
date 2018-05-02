#pragma once
#include "miss_program.h"
#include "environment_map.h"
#include "immediate_gui.h"
#include "texture.h"
#include <memory>
#include "optix_serialize.h"

class EnvironmentMap : public MissProgram
{
public:
    EnvironmentMap(std::string envmap_file = "") : envmap_path(envmap_file),
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
    std::unique_ptr<Texture> environment_sampler = nullptr;
    optix::Context context;
    optix::Buffer property_buffer;
    optix::Buffer sampling_property_buffer;
    int camera_1, camera_2, camera_3;
	std::string envmap_path;

    void presample_environment_map();
    bool resample_envmaps = true;

	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		archive( cereal::virtual_base_class<MissProgram>(this), CEREAL_NVP(environment_sampler));
	}

};

CEREAL_REGISTER_TYPE(EnvironmentMap)