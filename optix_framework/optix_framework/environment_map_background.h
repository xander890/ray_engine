#pragma once
#include "miss_program.h"
#include "environment_map.h"
#include "immediate_gui.h"
#include "texture.h"
#include <memory>
#include "optix_serialize_utils.h"

class EnvironmentMap : public MissProgram
{
public:
    EnvironmentMap(optix::Context & ctx, std::string envmap_file = "");

    virtual ~EnvironmentMap();

    virtual void init() override;
    virtual void load() override;
    virtual bool on_draw() override;
private:
    virtual bool get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program) override;
    
    EnvmapProperties properties;
    EnvmapImportanceSamplingData sampling_properties;

    optix::float3 envmap_deltas;
    std::unique_ptr<Texture> environment_sampler = nullptr;
    optix::Context mContext;
    optix::Buffer property_buffer;
    optix::Buffer sampling_property_buffer;

    unsigned int camera_1, camera_2, camera_3;
	std::string envmap_path = "";

    void presample_environment_map();
    bool resample_envmaps = true;

	friend class cereal::access;

	static void load_and_construct(cereal::XMLInputArchiveOptix & archive, cereal::construct<EnvironmentMap>& construct)
	{
		optix::Context ctx = archive.get_context();
		construct(ctx);
		archive(
			cereal::virtual_base_class<MissProgram>(construct.ptr()),
			cereal::make_nvp("texture", construct->environment_sampler),
			cereal::make_nvp("delta_rotation", construct->envmap_deltas),
			cereal::make_nvp("light_multiplier", construct->properties.lightmap_multiplier),
			cereal::make_nvp("importance_sample", construct->properties.importance_sample_envmap)
		);
	}

	template<class Archive>
	void save(Archive & archive) const
	{
		archive(
			 cereal::virtual_base_class<MissProgram>(this),
	    	 cereal::make_nvp("texture", environment_sampler),
			 cereal::make_nvp("delta_rotation", envmap_deltas),
             cereal::make_nvp("light_multiplier", properties.lightmap_multiplier),
             cereal::make_nvp("importance_sample", properties.importance_sample_envmap)
		);
	}

};

CEREAL_CLASS_VERSION(EnvironmentMap, 0)
CEREAL_REGISTER_TYPE(EnvironmentMap)