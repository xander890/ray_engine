/*
 *	Box class. This class represent a particular case of convex hull (6 planes) where all the planes form a box cube. The cube is axis aligned (use a transform to change its orientation). Center and size can be adjusted.
 */
#pragma once
#include "chull.h"

class Box :	public ConvexHull
{
public:
	Box(optix::Context ctx,  optix::float3 min_val,  optix::float3 max_val) : ConvexHull(ctx), mMinVal(fminf(min_val, max_val)), mMaxVal(fmaxf(min_val, max_val))
	{		
	}

	Box(optix::Context ctx,  optix::float3 center, float sidex ,float sidey, float sidez) : ConvexHull(ctx)
	{
		optix::float3 size_vec = 0.5f * optix::make_float3(sidex, sidey, sidez);
		mMinVal = center - size_vec;
		mMaxVal = center + size_vec;
	}

	bool on_draw() override;

private:
    Box() : ConvexHull() {}
	void make_planes(std::vector<Plane>& planes, optix::Aabb & bbox) override;

	// Serialization functions
    friend class cereal::access;
    void load( cereal::XMLInputArchiveOptix & archive)
    {
        archive(cereal::virtual_base_class<ConvexHull>(this));
        archive(cereal::make_nvp("min", mMinVal));
        archive(cereal::make_nvp("max", mMaxVal));
        ConvexHull::load();
    }

    template<class Archive>
    void save(Archive & archive) const
    {
        archive(cereal::virtual_base_class<ConvexHull>(this));
        archive(cereal::make_nvp("min", mMinVal));
        archive(cereal::make_nvp("max", mMaxVal));
    }

	 optix::float3 mMinVal, mMaxVal;

};

CEREAL_CLASS_VERSION(Box, 0)
CEREAL_REGISTER_TYPE(Box)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ConvexHull, Box)

