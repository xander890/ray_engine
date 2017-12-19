
#ifndef chull_h__
#define chull_h__

#include <optixu/optixu_aabb.h>
#include <optixu/optixpp_namespace.h>
#include <vector>
#include "procedural_mesh.h"

struct Plane
{
	float3 point;
	float3 normal;
};

class ConvexHull : public ProceduralMesh
{
public:
	ConvexHull() : ProceduralMesh() {}
	static const std::string id;
	
protected:
	virtual void init();
	virtual void make_planes(std::vector<Plane>& planes, optix::Aabb & bbox) = 0;


};

#endif // chull_h__
