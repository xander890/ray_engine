#include "procedural_loader.h"
#include <fstream>
#include <sstream>
#include "glm.h"
#include "scattering_material.h"

using namespace std;
using namespace optix;



ProceduralLoader::ProceduralLoader(ProceduralMesh * aobject, optix::Context& context, /* Context for RT object creation */ optix::GeometryGroup geometrygroup /* Empty geom group to hold model */)
	: ObjLoader("", context, geometrygroup, 0),
	object(aobject)
{

}

void ProceduralLoader::setIntersectProgram(optix::Program program)
{
	object->set_intersect_program(program);
}

void ProceduralLoader::setBboxProgram(optix::Program program)
{
	object->set_intersect_program(program);
}

std::vector<std::unique_ptr<Mesh>> ProceduralLoader::load()
{
	return load(optix::Matrix4x4::identity());
}

std::vector<std::unique_ptr<Mesh>> ProceduralLoader::load(const optix::Matrix4x4& transform)
{
	// We fake an obj file and save it in order to use the same pipeline for obj material files.
	char* fake_obj = "mtllib %s\n"
		"g Empty\n"
		"v 0.0 0.0 0.0\n"
		"v 0.0 0.0 0.0\n"
		"v 0.0 0.0 0.0\n"
		"usemtl %s\n"
		"f 1 2 3\n";
	char buffer[256];
	sprintf_s(buffer, fake_obj, object->get_material_file().c_str(), object->get_material().c_str());
	std::ofstream outfile(Folders::data_folder + "./procedural/placeholder.obj");
	outfile << buffer;
	outfile.close();

	GLMmodel* model = glmReadOBJ((Folders::data_folder + "./procedural/placeholder.obj").c_str());
	if (!model) {
		std::stringstream ss;
		ss << "ObjLoader::loadImpl - glmReadOBJ( '" << m_filename << "' ) failed" << std::endl;
	}

	createMaterialParams(model);
	Material & material = m_context->createMaterial();
	object->set_context(m_context);
	GeometryInstance instance = m_context->createGeometryInstance(object->get_geometry(), &material, &material + 1);
	
	optix::Acceleration acceleration = m_context->createAcceleration("Sbvh", "Bvh");

	m_geometrygroup->setAcceleration(acceleration);
	acceleration->markDirty();

	// Set up group 
	const unsigned current_child_count = m_geometrygroup->getChildCount();
	m_geometrygroup->setChildCount(current_child_count + 1);
	m_geometrygroup->setChild(current_child_count, instance);
	getMaterial(model->groups[0].material);

	glmDelete(model);
    // FIXME
	std::vector<std::unique_ptr<Mesh>> r;
	r.push_back(std::make_unique<Mesh>(m_context));
    return r;

}

optix::Aabb ProceduralLoader::getSceneBBox() const
{
	return object->get_BBox();
}


void ProceduralLoader::getAreaLights(std::vector<TriangleLight> & lights)
{
	// Procedural objects cannot be emissive (there are no triangles).
	return;
}
