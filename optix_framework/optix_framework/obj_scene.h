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
#include <camera_host.h>
#include <chrono>
#include <sstream>
#include "scene.h"

class MissProgram;
class ImmediateGUI;
class RenderTask;
struct TriangleLight;
class Object;
class MaterialHost;
class RenderingMethod;
struct MPMLMedium;
class Camera;

class ObjScene : public SampleScene
{
public:

	ObjScene(const std::vector<std::string>& obj_filenames, optix::int4 rendering_r = optix::make_int4(-1));
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

	void initialize_scene(GLFWwindow * window) override;
	void trace() override;


	void collect_image(unsigned int frame) const;
	bool key_pressed(int key, int x, int y) override;
	optix::Buffer get_output_buffer() override;
	void do_resize(unsigned int width, unsigned int height) override;
	void resize(unsigned int width, unsigned int height) override;
	void post_draw_callback() override;
	void set_debug_pixel(unsigned int x, unsigned int y);
	bool mouse_pressed(int x, int y,int button, int action, int mods) override;
	bool mouse_moving(int x, int y) override;
	void reset_renderer();
	void start_render_task_on_scene_ready();
	void scene_initialized() override;

	std::unique_ptr<RenderTask> current_render_task;

	void set_render_task(std::unique_ptr<RenderTask>& task);

	std::string override_mat = "";
	void add_override_material_file(std::string mat);

    Camera* get_camera() override;

private:
	optix::Context context;
	bool debug_mode_enabled = true;

	bool collect_images = false;
	bool show_difference_image = false;
	optix::Aabb m_scene_bounding_box;
	optix::Buffer createPBOOutputBuffer(const char* name, RTformat format, RTbuffertype type, unsigned width, unsigned height);

	void add_lights();
	void set_miss_program(BackgroundType::EnumType miss_program);

	static bool export_raw(const std::string& name, optix::Buffer buffer, int frames);
	void set_rendering_method(RenderingMethodType::EnumType t);
	std::vector<std::string> filenames;

	unsigned int m_frame;
	optix::Buffer output_buffer;

	std::unique_ptr<ImmediateGUI> gui = nullptr;
	std::unique_ptr<Scene> mScene;
	std::shared_ptr<MaterialHost> material_ketchup;

	void setDebugEnabled(bool var);
	float tonemap_multiplier = 1.0f;
	float tonemap_exponent = 1.0f;

	std::vector<MPMLMedium*> available_media;

	optix::Buffer rendering_output_buffer;
	optix::Buffer tonemap_output_buffer;
	optix::Buffer debug_output_buffer;
	optix::Buffer returned_buffer;

	void load_camera_extrinsics();

	optix::int4 custom_rr;

	optix::uint4 zoom_debug_window = optix::make_uint4(20,20,300,300);
	optix::uint4 zoomed_area = optix::make_uint4(0);

	std::chrono::high_resolution_clock::duration render_time_main = std::chrono::high_resolution_clock::duration::zero();
    std::chrono::high_resolution_clock::duration render_time_pre_trace = std::chrono::high_resolution_clock::duration::zero();
    std::chrono::high_resolution_clock::duration render_time_post = std::chrono::high_resolution_clock::duration::zero();
    std::chrono::high_resolution_clock::duration render_time_tonemap = std::chrono::high_resolution_clock::duration::zero();

	bool mbPaused = false;

	bool mImportanceSampleAreaLights = false;

	unsigned int tonemap_entry_point;
	unsigned int debug_entry_point;

	bool start_render_task_when_ready = false;

    void serialize_scene();
};

#endif // OBJSCENE_H
