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

struct RayTracerParameters
{
    size_t print_buffer_size = 2000;
    optix::int2 print_index = optix::make_int2(407, 56);
    bool debug_enabled = false;
    size_t stack_size = 30000;
    optix::float3 exception_color = optix::make_float3(0,0,1);
    int max_depth = 10;
    float scene_epsilon_fraction = 1e-4f;
    bool use_auto_camera = true;
};

struct TonemapParameters
{
	float exponent = 1.8f, multiplier = 1.0f;
};

class ObjScene : public SampleScene
{
public:

	ObjScene(const std::vector<std::string>& filenames);
	ObjScene();

	virtual ~ObjScene();

	void clean_up() override
	{
		context->destroy();
		m_context = nullptr;
	}

	bool draw_gui();

    void create_3d_noise(float frequency);

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

    Camera* get_camera() override;

private:
	optix::Context context;

	bool collect_images = false;
	bool show_difference_image = false;
	optix::Aabb m_scene_bounding_box;
	optix::Buffer createPBOOutputBuffer(const char* name, RTformat format, RTbuffertype type, unsigned width, unsigned height);

	static bool export_raw(const std::string& name, optix::Buffer buffer, int frames);
	void set_rendering_method(RenderingMethodType::EnumType t);
	std::vector<std::string> filenames;

	unsigned int m_frame;
	optix::Buffer output_buffer;

	std::unique_ptr<ImmediateGUI> gui = nullptr;
	std::unique_ptr<Scene> mScene;
	std::unique_ptr<Scene> mNewScene;

	void setDebugEnabled(bool var);

	std::vector<MPMLMedium*> available_media;

	optix::Buffer rendering_output_buffer;
	optix::Buffer tonemap_output_buffer;
	optix::Buffer debug_output_buffer;
	optix::Buffer returned_buffer;

	void load_default_camera();

	optix::uint4 zoom_debug_window = optix::make_uint4(20,20,300,300);
	optix::uint4 zoomed_area = optix::make_uint4(0);

	std::chrono::high_resolution_clock::duration render_time_main = std::chrono::high_resolution_clock::duration::zero();
    std::chrono::high_resolution_clock::duration render_time_pre_trace = std::chrono::high_resolution_clock::duration::zero();
    std::chrono::high_resolution_clock::duration render_time_post = std::chrono::high_resolution_clock::duration::zero();
    std::chrono::high_resolution_clock::duration render_time_tonemap = std::chrono::high_resolution_clock::duration::zero();

    std::chrono::time_point<std::chrono::high_resolution_clock> start_task_render_time;

	bool mbPaused = false;

	bool mImportanceSampleAreaLights = false;

	unsigned int tonemap_entry_point;
	unsigned int debug_entry_point;

	bool start_render_task_when_ready = false;

    RayTracerParameters parameters;
	TonemapParameters tonemap_parameters;
    void serialize_scene(const std::string &dest);

	void save_parameters(const std::string & config_file);
	void load_parameters(const std::string & config_file);

    void load_scene(const std::string &basic_string);
};

#endif // OBJSCENE_H
