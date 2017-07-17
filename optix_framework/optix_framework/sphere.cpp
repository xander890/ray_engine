#include "sphere.h"
#include "folders.h"

void Sphere::init()
{
	intersect_program = context->createProgramFromPTXFile(get_path_ptx("sphere.cu"), "sphere_intersect");
	bbox_program = context->createProgramFromPTXFile(get_path_ptx("sphere.cu"), "sphere_bounds");

	ProceduralMesh::init();

	geometry["center"]->setFloat(center);
	geometry["radius"]->setFloat(radius);
}

const std::string Sphere::id = "Sphere";

ProceduralMesh* Sphere::create(std::istream& i)
{
	float3 center;
	float radius;
	i >> center.x;
	i >> center.y;
	i >> center.z;
	i >> radius;
	return new Sphere(center, radius);
}

void Sphere::serialize(std::ostream& o) const
{
	o << id << " " << material << " " << center.x << " " << center.y << " " << center.z << " " << radius;
}