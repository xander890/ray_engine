#ifndef plint_h__
#define plint_h__

#pragma once
#include "chull.h"

class Plint : public ConvexHull
{
public:
	Plint(float3 center, float height, float bottom_base, float top_base) : ConvexHull(), height(height), bottom_base(bottom_base), top_base(top_base), center(center) {}

	virtual void make_planes(std::vector<Plane>& planes, optix::Aabb & bPlint) override;

	float3 center;
	float height, bottom_base, top_base;

	static const std::string id;
	static ProceduralMesh* create(std::istream&); 
	void serialize(std::ostream&) const override;
};


#endif // plint_h__
