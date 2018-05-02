#pragma once
#include "SampleScene.h"
#include <memory>
#include <optix_world.h>
#include "optix_serialize.h"
#include "singular_light.h"

class SingularLight
{
public:
    SingularLight();
    virtual void init(optix::Context & context);
    virtual SingularLightData get_data();
    virtual bool on_draw();
    virtual bool has_changed();

private:
    friend class cereal::access;

    void save(cereal::XMLOutputArchiveOptix & archive) const
    {
        archive(cereal::make_nvp("direction", mData->direction), cereal::make_nvp("type", mData->type), cereal::make_nvp("emission", mData->emission), cereal::make_nvp("casts_shadows", mData->casts_shadow));
    }

    void load(cereal::XMLInputArchiveOptix & archive)
    {
        mContext = archive.get_context();
        archive(cereal::make_nvp("direction", mData->direction), cereal::make_nvp("type", mData->type), cereal::make_nvp("emission", mData->emission), cereal::make_nvp("casts_shadows", mData->casts_shadow));
        init(mContext);
    }

    optix::Context mContext;
    std::unique_ptr<SingularLightData> mData = nullptr;
    bool mHasChanged = true;
    optix::float3 mColor;
    float mIntensity;
};
