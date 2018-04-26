
#include <optix_world.h>
#include "light_host.h"
#include "immediate_gui.h"



void SingularLight::init(optix::Context &context)
{
    mContext = context;
    mData = std::make_unique<SingularLightData>();
    mColor = normalize(mData->emission);
    mIntensity = length(mData->emission);
}

bool SingularLight::on_draw()
{
    std::string options = LightType::get_option_string();
    mHasChanged |= ImmediateGUIDraw::Combo("Type", (int*)&mData->type, options.c_str(), LightType::count());
    const char * string_pos = mData->type == LightType::DIRECTIONAL? "Direction" : "Position";
    mHasChanged |= ImmediateGUIDraw::InputFloat3(string_pos, &mData->direction.x);
    mHasChanged |= ImmediateGUIDraw::ColorEdit3("Color", &mColor.x);
    mHasChanged |= ImmediateGUIDraw::InputFloat("Intensity", &mIntensity);
    mHasChanged |= ImmediateGUIDraw::Checkbox("Use shadows", (bool*)&mData->casts_shadow);
    return mHasChanged;
}

SingularLightData SingularLight::get_data()
{
    mData->emission = mColor * mIntensity;
    return *mData;
}

bool SingularLight::has_changed()
{
    return mHasChanged;
}

