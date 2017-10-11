// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#ifndef OBJSCENE_H
#define OBJSCENE_H

#include <memory>
#include <string>
#include <vector>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixpp_namespace.h>
#include <SampleScene.h>
#include "enums.h"
#include <functional>

class MissProgram;
class ImmediateGUI;
class RenderTask;
struct TriangleLight;
class Mesh;
class MaterialHost;
class RenderingMethod;
class Camera;
struct MPMLMedium;


class ObjScene : public SampleScene
{
public:

	ObjScene(const std::vector<std::string>& obj_filenames, const std::string& shader_name, const std::string& config_file, optix::int4 rendering_r = optix::make_int4(-1));
	ObjScene();

	virtual ~ObjScene();

	void clean_up() override
	{
		context->destroy();
		m_context = nullptr;
	}

	bool draw_gui();

    void create_3d_noise(float frequency);
    float noise_frequency = 25;
	int use_heterogenous_materials = 0;

	void initialize_scene(GLFWwindow * window,InitialCameraData& camera_data) override;
	void trace(const RayGenCameraData& camera_data, bool& display) override;

	void trace(const RayGenCameraData& camera_data) override
	{
		bool display = true;
		trace(camera_data, display);
	}

	void collect_image(unsigned int frame) const;
	bool key_pressed(int key, int x, int y) override;
	optix::Buffer get_output_buffer() override;
	void do_resize(unsigned int width, unsigned int height) override;
	void resize(unsigned int width, unsigned int height) override;
	void post_draw_callback() override;
	void set_debug_pixel(int i, int y);
	bool mouse_pressed(int x, int y,int button, int action, int mods) override;
	bool mouse_moving(int x, int y) override;
	void reset_renderer();
	void start_render_task_on_scene_ready();
	void scene_initialized() override;

	std::unique_ptr<RenderTask> current_render_task;

	void set_render_task(std::unique_ptr<RenderTask>& task);

	std::string override_mat = "";
	void add_override_material_file(std::string mat);
	void add_override_parameters(std::vector<std::string> & params);
private:
	optix::Context context;
	bool debug_mode_enabled = false;

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
	std::unique_ptr<RenderingMethod> method;

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
	double render_time_post = 0.0;
	double render_time_tonemap = 0.0;

	bool mbPaused = false;

	void transform_changed();
	bool mImportanceSampleAreaLights = true;

	unsigned int tonemap_entry_point;
	unsigned int debug_entry_point;

	bool start_render_task_when_ready = false;
	std::stringstream console_log;
};

#endif // OBJSCENE_H
