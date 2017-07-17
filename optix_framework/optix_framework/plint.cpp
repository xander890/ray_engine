#include "plint.h"
#include <optix_math.h>
using namespace optix;

void Plint::make_planes(std::vector<Plane>& planes, optix::Aabb & bPlint)
{
	float3 height_vec = make_float3(0.0f, height, 0.0f);

	planes.push_back({center + height_vec, normalize(height_vec) }); // top
	planes.push_back({ center, -normalize(height_vec) });			 // bottom


	float3 top_center = center + height_vec;
	float slope = height / (bottom_base - top_base) * 2;
	float top_half_base = top_base / 2.0f;
	float3 pyramid_offset = slope * normalize(height_vec) * top_half_base;
	float3 tang_px = normalize(pyramid_offset - make_float3(top_half_base,0.0f,0.0f));
	float3 tang_pz = normalize(pyramid_offset - make_float3(0.0f, 0.0f,top_half_base));


	float3 normal_px = make_float3(tang_px.y, -tang_px.x, 0.0f);
	float3 normal_mx = make_float3(-normal_px.x, normal_px.y,normal_px.z);
	float3 normal_pz = make_float3(0.0f, -tang_pz.z, tang_pz.y);
	float3 normal_mz = make_float3(normal_pz.x, normal_pz.y, -normal_pz.z);

	float half_base = (bottom_base) / 2.0f;
	planes.push_back({ center + make_float3(half_base, 0.0f, 0.0f), normal_px }); //+x
	planes.push_back({ center + make_float3(-half_base, 0.0f, 0.0f), normal_mx }); //-x
	planes.push_back({ center + make_float3(0.0f, 0.0f, half_base),  normal_pz }); //+z
	planes.push_back({ center + make_float3(0.0f, 0.0f, -half_base), normal_mz }); //-z

	bPlint.m_min = center - 0.5f * make_float3(bottom_base, 0.0f, bottom_base);
	bPlint.m_max = top_center + 0.5f * make_float3(bottom_base, 0.0f, bottom_base);
}


const std::string Plint::id = "Plint";

ProceduralMesh* Plint::create(std::istream& i)
{
	float3 center;
	float height, bottom_base, top_base;
	i >> center.x;
	i >> center.y;
	i >> center.z;
	i >> height;
	i >> bottom_base;
	i >> top_base;
	return new Plint(center, height, bottom_base, top_base);
}

void Plint::serialize(std::ostream& o) const
{
	o << id << " " << material << " "<< center.x << " " << center.y << " " << center.z << " " << height << " " << bottom_base << " " << top_base;
}