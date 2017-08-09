#pragma once
#include "obj_loader.h"
#include "procedural_mesh.h"
/**
*	Procedural object loader: instead of loading objects from a triangle mesh, loads them from a ConvexChull and a 
*	material file.
*/

class ProceduralLoader : public ObjLoader
{
public:

	ProceduralLoader(
		ProceduralMesh * object,
		optix::Context& context,               // Context for RT object creation
		optix::GeometryGroup geometrygroup   // Empty geom group to hold model
		);           

	~ProceduralLoader() {}

	virtual void setIntersectProgram(optix::Program program) override;
	virtual void setBboxProgram(optix::Program program) override;
	virtual std::vector<Mesh> load() override;
	virtual std::vector<Mesh> load(const optix::Matrix4x4& transform) override;
	virtual optix::Aabb getSceneBBox() const override;
	virtual void getAreaLights(std::vector<TriangleLight> & lights) override;

	ProceduralMesh * object;

};

