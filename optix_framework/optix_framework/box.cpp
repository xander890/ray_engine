#include <immediate_gui.h>
#include "box.h"

void Box::make_planes(std::vector<Plane>& planes, optix::Aabb & bbox)
{
	bbox.m_min = mMinVal;
	bbox.m_max = mMaxVal;

	float3 center = (mMinVal + mMaxVal) * 0.5f;

	planes.push_back({ make_float3(mMaxVal.x, center.y, center.z), make_float3( 1.0f,  0.0f,  0.0f) });
	planes.push_back({ make_float3(mMinVal.x, center.y, center.z), make_float3(-1.0f,  0.0f,  0.0f) });
	planes.push_back({ make_float3(center.x, mMaxVal.y, center.z), make_float3(0.0f, 1.0f, 0.0f) });
	planes.push_back({ make_float3(center.x, mMinVal.y, center.z), make_float3(0.0f, -1.0f, 0.0f) });
	planes.push_back({ make_float3(center.x, center.y, mMaxVal.z), make_float3(0.0f, 0.0f, 1.0f) });
	planes.push_back({ make_float3(center.x, center.y, mMinVal.z), make_float3(0.0f, 0.0f, -1.0f) });
}

bool Box::on_draw()
{
    mReloadGeometry |= ImmediateGUIDraw::InputFloat3("Min", &mMinVal.x);
    mReloadGeometry |= ImmediateGUIDraw::InputFloat3("Max", &mMaxVal.x);
    return mReloadGeometry;
}
