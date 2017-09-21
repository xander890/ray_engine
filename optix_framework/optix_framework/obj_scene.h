// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#ifndef OBJSCENE_H
#define OBJSCENE_H

#include <string>
#include <vector>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixpp_namespace.h>
#include <SampleScene.h>
#include "rendering_method.h"
#include "parameter_parser.h"
#include "sky_model.h"
#include "area_light.h"
#include "mesh.h"
#include "enums.h"
#include "immediate_gui.h"

class RenderTask;

#include "structs.h"
#include <functional>
#include "camera.h"
#include "camera_host.h"

class ObjScene : public SampleScene
{
public:

	ObjScene(const std::vector<std::string>& obj_filenames, const std::string& shader_name, const std::string& config_file, optix::int4 rendering_r = optix::make_int4(-1));
	ObjScene();

	virtual ~ObjScene();

	virtual void cleanUp()
	{
		context->destroy();
		m_context = 0;
	}

	bool drawGUI();

    void create_3d_noise(float frequency);
    float noise_frequency = 25;
	int use_heterogenous_materials = 0;

	virtual void initScene(GLFWwindow * window,InitialCameraData& camera_data) override;
	virtual void trace(const RayGenCameraData& camera_data, bool& display) override;

	virtual void trace(const RayGenCameraData& camera_data) override
	{
		bool display = true;
		trace(camera_data, display);
	}

	void collect_image(unsigned int frame);
	virtual bool keyPressed(int key, int x, int y) override;
	virtual optix::Buffer getOutputBuffer() override;
	virtual void doResize(unsigned int width, unsigned int height) override;
	virtual void resize(unsigned int width, unsigned int height) override;
	void postDrawCallBack() override;
	void setDebugPixel(int i, int y);
	bool mousePressed(int x, int y,int button, int action, int mods) override;
	bool mouseMoving(int x, int y) override;
	void reset_renderer();
	void start_render_task();

	std::unique_ptr<RenderTask> current_render_task;

	void set_render_task(std::unique_ptr<RenderTask>& task);

	std::string override_mat = "";
	void add_override_material_file(std::string mat);
	void add_override_parameters(std::vector<std::string> & params);
private:
	optix::Context context;
	bool debug_mode_enabled = true;

	Scene::EnumType current_scene_type;

	BackgroundType::EnumType current_miss_program;
	bool collect_images = false;
	bool show_difference_image = false;
	optix::Aabb m_scene_bounding_box;
	optix::Buffer createPBOOutputBuffer(const char* name, RTformat format, RTbuffertype type, unsigned width, unsigned height);

	void add_lights(std::vector<TriangleLight>& area_lights);
	void set_miss_program();
    std::unique_ptr<MissProgram> miss_program = nullptr;

	// Geometry and transformation getters
	optix::GeometryGroup get_geometry_group(unsigned int idx);
	optix::Matrix4x4 get_object_transform(std::string filename);

	static bool export_raw(const std::string& name, optix::Buffer buffer, int frames);
	void set_rendering_method(RenderingMethodType::EnumType t);
	std::vector<std::string> filenames;

	optix::Group scene;
	//std::vector<optix::uint2> lights;
	RenderingMethod* method;

	unsigned int m_frame;
	bool deforming;
	std::unique_ptr<Camera> camera = nullptr;
	optix::Buffer output_buffer;
	
   
	std::unique_ptr<ImmediateGUI> gui = nullptr;
	void add_result_image(const std::string& image_file);
    std::vector<std::unique_ptr<Mesh>> mMeshes;
    std::shared_ptr<MaterialHost> material_ketchup;

    void execute_on_scene_elements(std::function<void(Mesh&)> operation);

	void setDebugEnabled(bool var);
	float tonemap_multiplier = 1.0f;
	float tonemap_exponent = 1.0f;

	optix::float3 global_absorption_override = optix::make_float3(0.0f);
	float global_absorption_inv_multiplier = 1.0f;
	float global_ior_override = 1.5f;
	float calc_absorption[3];
	void updateGlassObjects();
	std::vector<MPMLMedium*> available_media;
	int current_medium = 0;

	optix::Buffer rendering_output_buffer;
	optix::Buffer tonemap_output_buffer;
	optix::Buffer debug_output_buffer;
	optix::Buffer returned_buffer;

	int mtx_method = 1;
	
	optix::TextureSampler comparison_image;
	float comparison_image_weight = 0.0;

	void load_camera_extrinsics(InitialCameraData & data);

	const std::string config_file = "config.xml";
	optix::int4 custom_rr;

	optix::uint4 zoom_debug_window = optix::make_uint4(20,20,300,300);
	optix::uint4 zoomed_area = optix::make_uint4(0);

	std::vector<std::string> parameters_override;

	double render_time_main = 0.0;
	double render_time_load = 0.0;
	double render_time_pre_trace = 0.0;
	double render_time_tonemap = 0.0;

	bool mbPaused = false;

	void transform_changed();
	bool mImportanceSampleAreaLights = true;
};

#endif // OBJSCENE_H
