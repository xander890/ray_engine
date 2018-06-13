#pragma once

#include <optixu/optixu_aabb.h>
#include <optixu/optixpp_namespace.h>
#include <vector>
#include "geometry.h"

class Sphere : public Geometry
{
public:
    Sphere(optix::Context ctx, float3 center, float radius) : Geometry(ctx), center(center), radius(radius)
    {
    }

	void load() override;
	bool on_draw() override;

protected:
	void create_and_bind_optix_data() override;

	float3 center;
	float radius;

	optix::Aabb mBoundingBox;

private:
    Sphere() : Geometry() {}
	friend class cereal::access;
	// Serialization
	void load( cereal::XMLInputArchiveOptix & archive)
	{
		archive(cereal::virtual_base_class<Geometry>(this));
		archive(cereal::make_nvp("radius", radius));
		archive(cereal::make_nvp("center", center));
		load();
	}

	template<class Archive>
	void save(Archive & archive) const
	{
		archive(cereal::virtual_base_class<Geometry>(this));
		archive(cereal::make_nvp("radius", radius));
		archive(cereal::make_nvp("center", center));
	}
};

CEREAL_CLASS_VERSION(Sphere, 0)
CEREAL_REGISTER_TYPE(Sphere)
CEREAL_REGISTER_POLYMORPHIC_RELATION(Geometry, Sphere)
