#pragma once
#include "rendering_method.h"


class PathTracing :	public RenderingMethod
{
public:
	PathTracing() : RenderingMethod() {}

    void init(optix::Context &ctx) override;

    void pre_trace() override;
	std::string get_suffix() const { return "_path_tracing"; }

private:
    friend class cereal::access;

    template<class Archive>
    void serialize(Archive & archive)
    {
        archive(CEREAL_NVP(N));
    }
    unsigned int N;
};

CEREAL_REGISTER_TYPE(PathTracing)
CEREAL_REGISTER_POLYMORPHIC_RELATION(RenderingMethod, PathTracing)
