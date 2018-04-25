
#include "SampleScene.h"
#include <memory>
#include <optix_world.h>
#include "optix_serialize.h"
#include "singular_light.h"

class SingularLight
{
public:
    virtual void init(optix::Context & context);
    virtual void set_into_gpu();
    virtual bool on_draw();

private:
    friend class cereal::access;
    template<typename Archive>

    void serialize(Archive & archive)
    {
        archive(cereal::make_nvp("direction", mData->direction), cereal::make_nvp("type", mData->type), cereal::make_nvp("emission", mData->emission), cereal::make_nvp("casts_shadows", mData->casts_shadow));
    }

    optix::Context mContext;
    std::unique_ptr<SingularLightData> mData = nullptr;
};
