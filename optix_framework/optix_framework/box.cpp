#include "box.h"

const std::string Box::id = "Box";

void Box::make_planes(std::vector<Plane>& planes, optix::Aabb & bbox)
{
	bbox.m_min = min_val;
	bbox.m_max = max_val;
	
	float3 center = (min_val + max_val) * 0.5f;

	planes.push_back({ make_float3(max_val.x, center.y, center.z), make_float3( 1.0f,  0.0f,  0.0f) });
	planes.push_back({ make_float3(min_val.x, center.y, center.z), make_float3(-1.0f,  0.0f,  0.0f) });
	planes.push_back({ make_float3(center.x, max_val.y, center.z), make_float3(0.0f, 1.0f, 0.0f) });
	planes.push_back({ make_float3(center.x, min_val.y, center.z), make_float3(0.0f, -1.0f, 0.0f) });
	planes.push_back({ make_float3(center.x, center.y, max_val.z), make_float3(0.0f, 0.0f, 1.0f) });
	planes.push_back({ make_float3(center.x, center.y, min_val.z), make_float3(0.0f, 0.0f, -1.0f) });
}


ProceduralMesh* Box::create(std::istream& i)
{
	float3 min_val;
	i >> min_val.x;
	i >> min_val.y;
	i >> min_val.z;
	float3 max_val;
	i >> max_val.x;
	i >> max_val.y;
	i >> max_val.z;
	return new Box(min_val, max_val);
}

void Box::serialize(std::ostream& o) const
{
	o << id << " " << material << " " << min_val.x << " " << min_val.y << " " << min_val.z;
	o << " " << max_val.x << " " << max_val.y << " " << max_val.z;
}