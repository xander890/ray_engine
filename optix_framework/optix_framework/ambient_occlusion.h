#pragma once
#include "rendering_method.h"

class AmbientOcclusion : public RenderingMethod
{
public:
	AmbientOcclusion() : RenderingMethod() {}

	virtual void init(optix::Context& context) override;

	virtual void pre_trace();
	std::string get_suffix() const { return "_ao"; }

private:
	friend class cereal::access;

	template<class Archive>
	void serialize(Archive & archive)
	{
	}
};

CEREAL_REGISTER_TYPE(AmbientOcclusion)
CEREAL_REGISTER_POLYMORPHIC_RELATION(RenderingMethod, AmbientOcclusion)
