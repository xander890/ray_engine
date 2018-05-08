#ifndef plint_h__
#define plint_h__

#pragma once
#include "chull.h"

class Plint : public ConvexHull
{
public:
	Plint(optix::Context ctx, float3 center, float height, float bottom_base, float top_base) : ConvexHull(ctx), height(height), bottom_base(bottom_base), top_base(top_base), center(center) {}

	virtual void make_planes(std::vector<Plane>& planes, optix::Aabb & bPlint) override;

    bool on_draw() override;
	float3 center;
	float height, bottom_base, top_base;

private:
    Plint() : ConvexHull() {}
	friend class cereal::access;
	// Serialization
	void load( cereal::XMLInputArchiveOptix & archive)
	{
		archive(cereal::virtual_base_class<ConvexHull>(this));
		archive(cereal::make_nvp("center", center));
		archive(cereal::make_nvp("height", height));
		archive(cereal::make_nvp("bottom_base", bottom_base));
		archive(cereal::make_nvp("top_base", top_base));
		ConvexHull::load();
	}

	template<class Archive>
	void save(Archive & archive) const
	{
		archive(cereal::virtual_base_class<ConvexHull>(this));
		archive(cereal::make_nvp("center", center));
		archive(cereal::make_nvp("height", height));
		archive(cereal::make_nvp("bottom_base", bottom_base));
		archive(cereal::make_nvp("top_base", top_base));
	}
};

CEREAL_REGISTER_TYPE(Plint)
CEREAL_REGISTER_POLYMORPHIC_RELATION(ConvexHull, Plint)


#endif // plint_h__
