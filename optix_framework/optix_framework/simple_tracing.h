#pragma once
#include "rendering_method.h"

class SimpleTracing : public RenderingMethod 
{
public:
    SimpleTracing() : RenderingMethod() {}
	void init(optix::Context & ctx) override;
	std::string get_suffix() const { return ""; }

private:
    friend class cereal::access;

    template<class Archive>
    void serialize(Archive & archive)
    {
    }
};

CEREAL_REGISTER_TYPE(SimpleTracing)
CEREAL_REGISTER_POLYMORPHIC_RELATION(RenderingMethod, SimpleTracing)
