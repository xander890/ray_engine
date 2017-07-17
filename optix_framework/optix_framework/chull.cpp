#include "chull.h"
#include "folders.h"

using namespace optix;
using namespace std;

void ConvexHull::init()
{
	intersect_program = context->createProgramFromPTXFile(get_path_ptx("chull.cu"), "chull_intersect");
	bbox_program = context->createProgramFromPTXFile(get_path_ptx("chull.cu"), "chull_bounds");

	ProceduralMesh::init();

	Buffer plane_buffer = context->createBuffer(RT_BUFFER_INPUT);
	vector<Plane> planes;
	make_planes(planes, bbox);
	int nsides = planes.size();

	plane_buffer->setFormat(RT_FORMAT_FLOAT4);
	plane_buffer->setSize(nsides);

	float4* chplane = (float4*)plane_buffer->map();

	for (int i = 0; i < nsides; i++){
		float3 p = planes[i].point;
		float3 n = normalize(planes[i].normal);
		chplane[i] = make_float4(n, -dot(n, p));
	}
	plane_buffer->unmap();
	geometry["planes"]->setBuffer(plane_buffer);
	geometry["chull_bbmin"]->setFloat(bbox.m_min);
	geometry["chull_bbmax"]->setFloat(bbox.m_max);
}

