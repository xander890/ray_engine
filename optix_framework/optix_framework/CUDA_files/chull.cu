
#include <device_common.h>
#include <optixu/optixu_aabb.h>
#include <float.h>

using namespace optix;

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float2, texcoord, attribute texcoord, );
//
// (NEW)
// Bounding box program for programmable convex hull primitive
//
rtDeclareVariable(float3, chull_bbmin, , );
rtDeclareVariable(float3, chull_bbmax, , );

RT_PROGRAM void chull_bounds(int primIdx, float result[6])
{
	optix::Aabb* aabb = (optix::Aabb*)result;
	aabb->m_min = chull_bbmin;
	aabb->m_max = chull_bbmax;
}


//
// (NEW)
// Intersection program for programmable convex hull primitive
//
rtBuffer<float4> planes;
RT_PROGRAM void chull_intersect(int primIdx)
{
	int n = planes.size();
	float t0 = -FLT_MAX;
	float t1 = FLT_MAX;
	float3 t0_normal = make_float3(0);
	float3 t1_normal = make_float3(0);
	for (int i = 0; i < n && t0 < t1; ++i) {
		float4 plane = planes[i];
		float3 n = make_float3(plane);
		float  d = plane.w;

		float denom = dot(n, ray.direction);
		float t = -(d + dot(n, ray.origin)) / denom;
		if (denom < 0){
			// enter
			if (t > t0){
				t0 = t;
				t0_normal = n;
			}
		}
		else {
			//exit
			if (t < t1){
				t1 = t;
				t1_normal = n;
			}
		}
	}

	if (t0 > t1)
		return;

	if (rtPotentialIntersection(t0)){
		shading_normal = geometric_normal = t0_normal;
		texcoord = make_float2(0.0f);
		rtReportIntersection(0);
	}
	else if (rtPotentialIntersection(t1)){
		shading_normal = geometric_normal = t1_normal;
		texcoord = make_float2(0.0f);
		rtReportIntersection(0);
	}
}
