#ifndef AI_SCENE_LOADER_H
#define AI_SCENE_LOADER_H


#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include <string>
#include <folders.h>
#include <area_light.h>

#include <sutil\SampleScene.h>

#pragma once
#include <sutil/glm.h>


class AISceneLoader
{
public:

	AISceneLoader(const char* filename,                 // Model filename
		optix::Context& context,               // Context for RT object creation
		const char* ASBuilder = "Sbvh",
		const char* ASTraverser = "Bvh",
		const char* ASRefine = "0",
		bool large_geom = false);

	~AISceneLoader() {} // makes sure CRT objects are destroyed on the correct heap

	virtual void setIntersectProgram(optix::Program program);
	virtual void setBboxProgram(optix::Program program);
	virtual void load();

	virtual optix::Aabb getSceneBBox()const { return m_aabb; }

	virtual void getAreaLights(std::vector<TriangleLight> & lights);

	struct MatParams
	{
		std::string name;
		optix::float3 emissive;
		optix::float3 reflectivity;
		float  phong_exp;
		float  ior;
		int    illum;
		optix::TextureSampler ambient_map;
		optix::TextureSampler diffuse_map;
		optix::TextureSampler specular_map;
	};

	std::vector<MatParams> getMaterialParameters()
	{
		return m_material_params;
	}



protected:

	void createMaterial();
    void createGeometryInstance(void* model, const optix::Matrix4x4& transform);
    void createMaterialParams(unsigned int index, void * mat);
	void loadMaterialParams(optix::GeometryInstance gi, unsigned int index);

	void createLightBuffer(GLMmodel* model, const optix::Matrix4x4& transform);

	void extractAreaLights(const GLMmodel * model, const optix::Matrix4x4 & transform, unsigned int & num_light, std::vector<TriangleLight> & lights);

	void get_camera_data(SampleScene::InitialCameraData & data);

	std::vector<TriangleLight> m_lights;
	std::string            m_pathname;
	std::string            m_filename;
	optix::Context         m_context;
	optix::GeometryGroup   m_geometrygroup;
	optix::Material        m_material;
	optix::Program         m_intersect_program;
	optix::Program         m_bbox_program;
	optix::Buffer          m_light_buffer;
	bool                   m_have_default_material;
	bool                   m_force_load_material_params;
	const char*            m_ASBuilder;
	const char*            m_ASTraverser;
	const char*            m_ASRefine;
	bool                   m_large_geom;
	optix::Aabb            m_aabb;
	std::vector<MatParams> m_material_params;
	SampleScene::InitialCameraData m_camera;
};

#endif

