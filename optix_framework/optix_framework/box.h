#ifndef box_h__
#define box_h__

#pragma once
#include "chull.h"

class Box :	public ConvexHull
{
public:
	Box(float3 min_val, float3 max_val) : ConvexHull(), max_val(fmaxf(min_val, max_val)), min_val(fminf(min_val, max_val)) {}
	Box(float3 center, float sidex ,float sidey, float sidez) : ConvexHull() 
	{
		float3 size_vec = 0.5f * make_float3(sidex, sidey, sidez);
		min_val = center - size_vec;
		max_val = center + size_vec;
	}

	virtual void make_planes(std::vector<Plane>& planes, optix::Aabb & bbox) override;

	float3 min_val, max_val;

	static const std::string id;
	static ProceduralMesh* create(std::istream&);
	void serialize(std::ostream&) const override;
};

#endif // box_h__
