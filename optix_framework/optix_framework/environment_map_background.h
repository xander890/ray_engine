#pragma once
#include "miss_program.h"
#include "environment_map.h"
#include "immediate_gui.h"
#include "texture.h"
#include <memory>
#include "optix_serialize_utils.h"

/*
 * Miss program describing an environment map.
 */
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
  
    EnvmapProperties mProperties;
    EnvmapImportanceSamplingData mSamplingProperties;

    optix::float3 mEnvmapRotationDeltas;
    std::unique_ptr<Texture> mEnvmapTexture = nullptr;

    unsigned int mEntryPoint1, mEntryPoint2, mEntryPoint3;
	std::string mEnvmapPath = "";

    void presample_environment_map();
    bool mResample = true;

    // Serialization
	friend class cereal::access;
    EnvironmentMap() = default;
    void load(cereal::XMLInputArchiveOptix& archive);
    void save(cereal::XMLOutputArchiveOptix& archive) const;
};

CEREAL_CLASS_VERSION(EnvironmentMap, 0)
CEREAL_REGISTER_TYPE(EnvironmentMap)
CEREAL_REGISTER_POLYMORPHIC_RELATION(MissProgram, EnvironmentMap)