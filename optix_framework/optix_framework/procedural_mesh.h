#ifndef procedural_mesh_h__
#define procedural_mesh_h__

#include <optixu\optixu_aabb.h>
#include <optixu\optixpp_namespace.h>
#include <vector>
#include <map>
#include <functional>

class ProceduralMesh
{
public:
    virtual ~ProceduralMesh() = default;

    typedef ProceduralMesh* (*Factory)(std::istream&);

	ProceduralMesh() : context(0), bbox(), geometry(), initialized(false), bbox_program(0), intersect_program(0), material_file("procedural_materials.mtl"), material("white") 
	{

	}
	optix::Aabb get_BBox();
	optix::Geometry& get_geometry();
	std::string get_material_file() { return material_file; }
	std::string get_material() { return material; }

	void set_intersect_program(optix::Program intersect){ intersect_program = intersect; }
	void set_bbox_program(optix::Program program) { bbox_program = program; }
	void set_context(optix::Context xcontext) { context = xcontext; }
	void set_material_file(const char* mmaterial_file) { material_file = std::string(mmaterial_file); }
	void set_material(const char* mmaterial) 
	{ material = std::string(mmaterial); }
    
	static std::map<std::string, Factory> serialization_map;
	virtual void serialize(std::ostream&) const = 0;
	static ProceduralMesh* unserialize(std::istream&);

protected:
	virtual void init();
	optix::Aabb bbox;
	optix::Geometry geometry;
	optix::Context context;
	optix::Program bbox_program;
	optix::Program intersect_program;
	std::string material_file;
	std::string material;
	bool initialized;
};
#endif // procedural_mesh_h__