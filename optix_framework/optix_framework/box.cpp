#include <immediate_gui.h>
#include "box.h"

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

bool Box::on_draw()
{
    mReloadGeometry |= ImmediateGUIDraw::InputFloat3("Min", &min_val.x);
    mReloadGeometry |= ImmediateGUIDraw::InputFloat3("Max", &max_val.x);
    return mReloadGeometry;
}
