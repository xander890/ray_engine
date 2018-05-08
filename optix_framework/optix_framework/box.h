#ifndef box_h__
#define box_h__

#pragma once
#include "chull.h"

class Box :	public ConvexHull
{
public:
	Box(optix::Context ctx, float3 min_val, float3 max_val) : ConvexHull(ctx), max_val(fmaxf(min_val, max_val)), min_val(fminf(min_val, max_val)) {}
	Box(optix::Context ctx, float3 center, float sidex ,float sidey, float sidez) : ConvexHull(ctx)
	{
		float3 size_vec = 0.5f * optix::make_float3(sidex, sidey, sidez);
		min_val = center - size_vec;
		max_val = center + size_vec;
	}

	virtual void make_planes(std::vector<Plane>& planes, optix::Aabb & bbox) override;

    bool on_draw() override;

	float3 min_val, max_val;

private:
    Box() : ConvexHull() {}

    friend class cereal::access;
    // Serialization
    void load( cereal::XMLInputArchiveOptix & archive)
    {
        archive(cereal::virtual_base_class<ConvexHull>(this));
        archive(cereal::make_nvp("min", min_val));
        archive(cereal::make_nvp("max", max_val));
        ConvexHull::load();
    }

    template<class Archive>
    void save(Archive & archive) const
    {
        archive(cereal::virtual_base_class<ConvexHull>(this));
        archive(cereal::make_nvp("min", min_val));
        archive(cereal::make_nvp("max", max_val));
    }

};

CEREAL_REGISTER_TYPE(Box)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ConvexHull, Box)

#endif // box_h__
