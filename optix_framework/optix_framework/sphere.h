#pragma once

#include <optixu\optixu_aabb.h>
#include <optixu\optixpp_namespace.h>
#include <vector>
#include "procedural_mesh.h"


class Sphere : public ProceduralMesh
{
public:
	Sphere(float3 center, float radius) : ProceduralMesh(), center(center), radius(radius) {}
	
	static const std::string id;
	static ProceduralMesh* create(std::istream&);
	void serialize(std::ostream&) const override;
protected:
	virtual void init();
	float3 center;
	float radius;


};