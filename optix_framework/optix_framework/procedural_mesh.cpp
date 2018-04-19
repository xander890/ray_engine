#include "procedural_mesh.h"
#include "folders.h"
#include "math_helpers.h"
#include "parserstringhelpers.h"
#include "box.h"
#include "plint.h"
#include "sphere.h"
#include "chull.h"
using namespace optix;
using namespace std;

std::map<std::string, ProceduralMesh::Factory> ProceduralMesh::serialization_map = { };

optix::Aabb ProceduralMesh::get_BBox()
{
	if (!initialized)
	{
		init();
	}
	return bbox;
}

optix::Geometry& ProceduralMesh::get_geometry()
{
	if (!initialized)
	{
		init();
	}
	return geometry;
}

void ProceduralMesh::init()
{
	if (initialized)
		return;
	initialized = true;

	if (!context->get())
	{	
		throw new Exception(string("No context provided in procedural mesh."));
		return;
	}

	geometry = context->createGeometry();
	geometry->setPrimitiveCount(1u);

	if (!intersect_program.get())
	{
		intersect_program = context->createProgramFromPTXFile(get_path_ptx("chull.cu"), "chull_intersect");
	}
	if (!bbox_program.get())
	{
		bbox_program = context->createProgramFromPTXFile(get_path_ptx("chull.cu"), "chull_bounds");
	}

	geometry->setBoundingBoxProgram(bbox_program);
	geometry->setIntersectionProgram(intersect_program); 
}

ProceduralMesh* ProceduralMesh::unserialize(std::istream& stream)
{
	if (serialization_map.size() == 0)
	{
		serialization_map[Box::id] = Box::create;
		serialization_map[Plint::id] = Plint::create;
		serialization_map[Sphere::id] = Sphere::create;
	}

	std::string name;
	stream >> name;
	std::string material;
	stream >> material;
	if (serialization_map.count(name) > 0)
	{
		auto m = serialization_map[name](stream);
		m->set_material(material.c_str());
		return m;
	}
	
	return nullptr;
}