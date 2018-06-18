#pragma once
#include "rendering_method.h"

/*
   Specific class implementation to render ambient occlusion only. Excludes color, any particular implementation particular element, showing only the scene with ambient occlusion. 

   Useful for debugging ray performance.

   WARNING: This method is mostly unimplemented in the current implementation.
*/
class AmbientOcclusion : public RenderingMethod
{
public:
	void init(optix::Context& context) override;
	std::string get_suffix() const override { return "_ao"; }

private:
	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
	}
};

CEREAL_REGISTER_TYPE(AmbientOcclusion)
CEREAL_CLASS_VERSION(AmbientOcclusion, 0)
CEREAL_REGISTER_POLYMORPHIC_RELATION(RenderingMethod, AmbientOcclusion)
