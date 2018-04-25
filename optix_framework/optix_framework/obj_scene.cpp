// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <cereal/archives/json.hpp>
#include <fstream>
#include "obj_loader.h"
#include "singular_light.h"
#include "obj_scene.h"
#include "simple_tracing.h"
#include "parameter_parser.h"
#include "material_library.h"
#include "ambient_occlusion.h"
#include "path_tracing.h"
#include "sphere.h"
#include "dialogs.h"
#include <ImageLoader.h>
#include "presampled_surface_bssrdf.h"
#include "GLFWDisplay.h"
#include "shader_factory.h"
#include "optprops/Medium.h"
#include "object_host.h"
#include "host_material.h"
#include "environment_map_background.h"
#include "constant_background.h"
#include "PerlinNoise.h"
#include <algorithm>
#include "optix_utils.h"
#include <sampled_bssrdf.h>
#include "render_task.h"
#include "volume_path_tracer.h"
#include "GLFW/glfw3.h"
#include "optix_serialize.h"
#include "scattering_material.h"
#include "math_helpers.h"
#include "cputimer.h"
#include "rendering_method.h"
#include "sky_model.h"
#include "area_light.h"
#include "immediate_gui.h"
#include "camera_host.h"
#include "structs.h"
#include "bssrdf_visualizer.h"

using namespace std;
using optix::uint2;
using optix::TextureSampler;
using optix::Program;
using optix::make_float3;
using optix::Acceleration;
using optix::Aabb;
using optix::Buffer;

void ObjScene::collect_image(unsigned int frame) const
{
	if (!collect_images) return;

	const std::string name = std::string("rendering_") + to_string(frame) + ".raw";
	export_raw(name, rendering_output_buffer, frame);
}

void ObjScene::reset_renderer()
{
	clear_buffer(debug_output_buffer);
	clear_buffer(rendering_output_buffer);
	clear_buffer(tonemap_output_buffer);
	m_frame = 0;
}

bool ObjScene::key_pressed(int key, int action, int modifier)
{
	if (key == GLFW_KEY_P)
	{
		mbPaused = !mbPaused;
		return true;
	}
	if (current_render_task->is_active())
		return false;
	if (gui->keyPressed(key, action, modifier) || key >= 48 && key <= 57) // numbers avoided
	{
		reset_renderer();
		return true;
	}
	switch (key)
	{
	case GLFW_KEY_E:
	{
		const std::string res = std::string("result_optix.raw");
		return export_raw(res, rendering_output_buffer, m_frame);
	}
	case GLFW_KEY_G:
	{
		gui->toggleVisibility();
		return true;
	}
	case GLFW_KEY_R:
	{
		Logger::info << "Reloading all shaders..." << std::endl;
        mScene->reload();
	}
	break;
	default: return false;
	}
	return false;
}


ObjScene::ObjScene(const std::vector<std::string>& obj_filenames, optix::int4 rendering_r)
	: context(m_context),
	filenames(obj_filenames), m_frame(0u)
{
	custom_rr = rendering_r;
	current_render_task = make_unique<RenderTaskFrames>(1000, "res.raw", false);
}

ObjScene::ObjScene()
	: context(m_context),
     filenames(1, "test.obj"),
	m_frame(0u)
{
	custom_rr = optix::make_int4(-1);
	current_render_task = make_unique<RenderTaskFrames>(1000, "res.raw", false);
}

inline ObjScene::~ObjScene()
{
	ConfigParameters::free();
	ObjScene::clean_up();
}

float to_milliseconds(std::chrono::high_resolution_clock::duration & dur)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count() / 1e6f;
}

bool ObjScene::draw_gui()
{
	bool changed = false;
	ImmediateGUIDraw::TextColored({ 255,0,0,1 }, "Rendering info ");
	std::stringstream ss;
	ss << "Current frame: " << to_string(m_frame) << std::endl;
	ss << "Time (pre trace):   " << to_string(to_milliseconds(render_time_pre_trace)) << " ms (" << to_string(1000.0 / to_milliseconds(render_time_pre_trace)) << " FPS)" << std::endl;
	ss << "Time (render):      " << to_string(to_milliseconds(render_time_main)) << " ms (" << to_string(1000.0 / to_milliseconds(render_time_main)) << " FPS)" << std::endl;
	ss << "Time (post trace):  " << to_string(to_milliseconds(render_time_post)) << " ms (" << to_string(1000.0 / to_milliseconds(render_time_post)) << " FPS)" << std::endl;
	ss << "Time (tonemap/dbg): " << to_string(to_milliseconds(render_time_tonemap)) << " ms (" << to_string(1000.0 / to_milliseconds(render_time_tonemap)) << " FPS)";
	ImmediateGUIDraw::Text("%s",ss.str().c_str());
	static bool debug = debug_mode_enabled;

	if(ImmediateGUIDraw::Button("Serialize"))
	{
		serialize_scene();
	}

	if (ImmediateGUIDraw::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImmediateGUIDraw::Checkbox("Debug mode", &debug))
		{
			changed = true;
			setDebugEnabled(debug);
			if (debug)
			{
				returned_buffer = debug_output_buffer;
			}
			else
			{
				returned_buffer = tonemap_output_buffer;
			}
		}

		ImmediateGUIDraw::SameLine();
		ImmediateGUIDraw::Checkbox("Pause", &mbPaused);

		static int depth = context["max_depth"]->getInt();
		if (ImmediateGUIDraw::InputInt("Maximum ray depth", &depth, 1, 10))
		{
			changed = true;
			context["max_depth"]->setInt(depth);
		}

		if (ImmediateGUIDraw::Button("Reset Camera"))
		{
			changed = true;
			InitialCameraData i;
			load_camera_extrinsics(i);
			GLFWDisplay::setCamera(i);
		}

		ImmediateGUIDraw::SameLine();
		if (ImmediateGUIDraw::Button("Reset Shaders"))
		{
            mScene->reload();
		}

		if (ImmediateGUIDraw::Button("Save RAW image"))
		{
			std::string filePath;
			if (Dialogs::saveFileDialog(filePath))
			{
				export_raw(filePath, rendering_output_buffer, m_frame);
			}
		}
		ImmediateGUIDraw::SameLine();
		if (ImmediateGUIDraw::Button("Save current configuration file"))
		{
			std::string filePath;
			if (Dialogs::saveFileDialog(filePath))
			{
				ConfigParameters::dump_used_parameters(filePath);
			}
		}

	}


	if (debug && ImmediateGUIDraw::CollapsingHeader("Debug"))
	{
		uint2 debug_pixel = context["debug_index"]->getUint2();
		if (ImmediateGUIDraw::InputInt2("Debug pixel", (int*)&debug_pixel))
		{
			set_debug_pixel(debug_pixel.x, debug_pixel.y);
		}

        if(ImmediateGUIDraw::Button("All pixels"))
        {
            m_context->setPrintLaunchIndex(-1,-1,-1);
        }

		if (ImmediateGUIDraw::Button("Center debug pixel"))
		{
			set_debug_pixel(zoomed_area.x + zoomed_area.z / 2, zoomed_area.y + zoomed_area.w / 2);
		}

		ImmediateGUIDraw::InputInt4("Zoom window", (int*)&zoom_debug_window);
		
		ImmediateGUIDraw::InputInt4("Zoom area", (int*)&zoomed_area);
    }


	if (ImmediateGUIDraw::CollapsingHeader("Tone mapping"))
	{
		changed |= ImmediateGUIDraw::SliderFloat("Multiplier##TonemapMultiplier", &tonemap_multiplier, 0.0f, 2.0f, "%.3f", 1.0f);
		changed |= ImmediateGUIDraw::SliderFloat("Exponent##TonemapExponent", &tonemap_exponent, 0.5f, 3.5f, "%.3f", 1.0f);
		if (ImmediateGUIDraw::Button("Reset##TonemapExponentMultiplierReset"))
		{
			changed = true;
			tonemap_exponent = 1.8f;
			tonemap_multiplier = 1.0f;
		}
	}
	
	if (ImmediateGUIDraw::CollapsingHeader("Sampling (generic)"))
	{
		if (ImmediateGUIDraw::Checkbox("Importance sample area lights", &mImportanceSampleAreaLights))
		{
			changed = true;
			context["importance_sample_area_lights"]->setUint(static_cast<unsigned int>(mImportanceSampleAreaLights));
		}
	}

    const char * miss_programs[3] = { "Constant Background", "Environment map", "Sky model"  };
    static int current_miss_program = 0; //FIXME
	if (ImmediateGUIDraw::Combo("Background", &current_miss_program, miss_programs, 3, 3))
	{
		changed = true;
        set_miss_program(static_cast<BackgroundType::Type>(current_miss_program));
	}

    changed = mScene->on_draw();

	if (ImmediateGUIDraw::CollapsingHeader("Heterogenous materials"))
	{
		ImmediateGUIDraw::Checkbox("Enable##EnableHeterogenousMaterials", (bool*)&use_heterogenous_materials);
		if (ImmediateGUIDraw::InputFloat("Noise frequency##HeterogenousNoseFreq", &noise_frequency, 0.1f, 1.0f))
		{
			changed = true;
			create_3d_noise(noise_frequency);
		}
	}

	const bool is_active = current_render_task->is_active();
	const int flag = is_active ? ImGuiTreeNodeFlags_DefaultOpen : 0;
	if (ImmediateGUIDraw::CollapsingHeader("Render tasks", flag))
	{
		static int current_item = 0;
		const char * items[2] = { "Frame based", "Time based" };
		if (!current_render_task->is_active())
		{
			if (ImmediateGUIDraw::Combo("Render task type", &current_item, items, 2))
			{
				if(current_item == 0)
					current_render_task = std::make_unique<RenderTaskFrames>(1000, current_render_task->get_destination_file(), false);
				else
					current_render_task = std::make_unique<RenderTaskTime>(10.0f, current_render_task->get_destination_file(), false);
			}
		}
		current_render_task->on_draw();

		if (!current_render_task->is_active() && ImmediateGUIDraw::Button("Start task"))
		{
			changed = true;
			current_render_task->start();
		}

		if (current_render_task->is_active() && ImmediateGUIDraw::Button("End task"))
		{
			current_render_task->end();
		}
	}

	return changed;
}

void ObjScene::serialize_scene()
{
    mScene->serialize();
}


void ObjScene::create_3d_noise(float frequency)
{
    static Texture tex(context);
    tex.set_size(256,256,256);

    static PerlinNoise p(1337);
    // Create buffer with single texel set to default_color
    float* buffer_data = static_cast<float*>(tex.map_data());

    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++)
            for (int k = 0; k < 256; k++)
            {
                int idx = 256 * 256 * i + 256 * j + k;
                buffer_data[idx] = (float)p.noise(i / (256.0f) * frequency, j / (256.0f) * frequency, k / (256.0f) * frequency);
            }
    tex.unmap_data();
    context["noise_tex"]->setInt(tex.get_id());
}

void ObjScene::initialize_scene(GLFWwindow * , InitialCameraData& init_camera_data)
{
	Logger::info << "Initializing scene." << endl;
	context->setPrintBufferSize(2000);
	setDebugEnabled(debug_mode_enabled);
	context->setPrintLaunchIndex(407,56);

    mScene = std::make_unique<Scene>(context);
	Folders::init();
	MaterialLibrary::load(Folders::mpml_file.c_str());
	ScatteringMaterial::initializeDefaultMaterials();

	if (override_mat.size() > 0)
	{
		auto v = ObjLoader::parse_mtl_file(override_mat, context);
		MaterialHost::set_default_material(v[0]);
	}

    auto camera_type = PinholeCameraType::to_enum(ConfigParameters::get_parameter<string>("camera", "camera_definition_type", PinholeCameraType::to_string(PinholeCameraType::EYE_LOOKAT_UP_VECTORS), std::string("Type of the camera. Types: ") + PinholeCameraType::get_full_string()));
    unsigned int camera_width = ConfigParameters::get_parameter<unsigned int>("camera", "window_width", 512, "The width of the window");
    unsigned int camera_height = ConfigParameters::get_parameter<unsigned int>("camera", "window_height", 512, "The height of the window");
    int downsampling = ConfigParameters::get_parameter<int>("camera", "camera_downsampling", 1, "");
    auto camerau = std::make_unique<Camera>(context, camera_type, camera_width, camera_height, downsampling, custom_rr);
    mScene->set_current_camera(std::move(camerau));

    std::shared_ptr<Camera> camera = mScene->get_current_camera();
    RenderingMethodType::EnumType t = RenderingMethodType::to_enum(ConfigParameters::get_parameter<string>("config", "rendering_type", RenderingMethodType::to_string(RenderingMethodType::RECURSIVE_RAY_TRACING), std::string("Rendering method. ") + RenderingMethodType::get_full_string()));
    set_rendering_method(t);


    ShaderFactory::init(context);
	ShaderInfo info = ShaderInfo(12, "volume_shader.cu", "Volume path tracer");
	ShaderFactory::add_shader(std::make_unique<VolumePathTracer>(info));

	ShaderInfo info3 = ShaderInfo(13, "volume_shader_heterogenous.cu", "Volume path tracer (het.)"); 
	ShaderFactory::add_shader(std::make_unique<VolumePathTracer>(info3));
	
	ShaderInfo info2 = ShaderInfo(14, "subsurface_scattering_sampled_default.cu", "Sampled BSSRDF");
	ShaderFactory::add_shader(std::make_unique<SampledBSSRDF>(info2));

	ShaderInfo info6 = ShaderInfo(22, "empty.cu", "BSSRDF Visualizer");
	ShaderFactory::add_shader(std::make_unique<BSSRDFPlaneRenderer>(info6, camera_width, camera_height));

    for (auto& kv : MaterialLibrary::media)
	{
		available_media.push_back(&kv.second);
	}

    BackgroundType::Type current_miss_program = BackgroundType::to_enum(ConfigParameters::get_parameter<string>("config", "default_miss_type", BackgroundType::to_string(BackgroundType::CONSTANT_BACKGROUND), std::string("Miss program. ") + BackgroundType::get_full_string()));

    tonemap_exponent = ConfigParameters::get_parameter<float>("tonemap", "tonemap_exponent", 1.8f, "Tonemap exponent");
    tonemap_multiplier = ConfigParameters::get_parameter<float>("tonemap", "tonemap_multiplier", 1.f, "Tonemap multiplier");

	// Setup context
	context->setRayTypeCount(RayType::count());
	context->setStackSize((RTsize)ConfigParameters::get_parameter<int>("config", "stack_size", 2000, "Allocated stack size for context"));
	context["use_heterogenous_materials"]->setInt(use_heterogenous_materials);

	// Constant colors
	context["bad_color"]->setFloat(0.0f, 1.0f, 0.0f);
    context["bg_color"]->setFloat(0.3f, 0.3f, 0.3f);
	
	bool use_abs = ConfigParameters::get_parameter<bool>("config", "use_absorption", true, "Use absorption in rendering.");
	Logger::debug << "Absorption is " << (use_abs ? "ON" : "OFF") << endl;

	// Tone mapping pass
	rendering_output_buffer = createPBOOutputBuffer("output_buffer", RT_FORMAT_FLOAT4, RT_BUFFER_INPUT_OUTPUT, camera->get_width(), camera->get_height());
	tonemap_output_buffer = createPBOOutputBuffer("tonemap_output_buffer", RT_FORMAT_UNSIGNED_BYTE4, RT_BUFFER_INPUT_OUTPUT, camera->get_width(), camera->get_height());
	debug_output_buffer = createPBOOutputBuffer("debug_output_buffer", RT_FORMAT_UNSIGNED_BYTE4, RT_BUFFER_OUTPUT, camera->get_width(), camera->get_height());
	returned_buffer = tonemap_output_buffer;

	// Create group for scene objects and float acceleration structure

	// We need the scene bounding box for placing the camera
	Aabb bbox(make_float3(-1,-1,-1), make_float3(1,1,1));
	

	// Load geometry from OBJ files into the group of scene objects
	vector<TriangleLight> lights(0);
	Logger::info << "Loading obj files..." << endl;
#ifdef NEW_SCENE
	AISceneLoader l = AISceneLoader(s.c_str(), context);
#endif
	m_scene_bounding_box = bbox;
	for (unsigned int i = 0; i < filenames.size(); ++i)
	{
		// Load OBJ scene
		Logger::info <<"Loading obj " << filenames[i]  << "..." <<endl;
		ObjLoader* loader = new ObjLoader((Folders::data_folder + filenames[i]).c_str(), context);
        vector<std::unique_ptr<Object>> v = loader->load();
		for (auto& c : v)
		{
            mScene->add_object(std::move(c));
		}

	    m_scene_bounding_box.include(loader->getSceneBBox());
		loader->getAreaLights(lights);
		// Set material shaders

		// Add geometry group to the group of scene objects
		
		delete loader;
	}
	
	context["importance_sample_area_lights"]->setUint(static_cast<unsigned int>(mImportanceSampleAreaLights));

	// Procedural objects
	vector<ProceduralMesh*> procedural;

	std::ifstream pscenefile(Folders::data_folder + "./procedural/procedural_scene.txt");
	if (pscenefile) // same as: if (myfile.good())
	{
		std::string line;
		while (getline(pscenefile, line)) // same as: while (getline( myfile, line ).good())
		{
			std::stringstream ss(line);
			procedural.push_back(ProceduralMesh::unserialize(ss));
		}
		pscenefile.close();
	}

	
//	for (int i = 0; i < procedural.size(); i++)
//	{
//		GeometryGroup geometry_group = context->createGeometryGroup();
//		ProceduralMesh* mesh = procedural[i];
//		if (mesh != nullptr)
//		{
//			ObjLoader* loader = new ProceduralLoader(mesh, context, geometry_group);
//			loader->load();
//			m_scene_bounding_box.include(loader->getSceneBBox());
//			loader->getAreaLights(lights);
//			delete loader;
//
//			// Set material shaders
//			for (unsigned int j = 0; j < geometry_group->getChildCount(); ++j)
//			{
////				GeometryInstance gi = geometry_group->getChild(j);
////				addMERLBRDFtoGeometry(gi, use_merl_brdf);
//
//                // FIXME
//				//method->init(gi);
//			}
//
//			// Add geometry group to the group of scene objects
//			scene->setChild(static_cast<unsigned int>(filenames.size()) + i, geometry_group);
//		}
//	}


	// Add light sources depending on chosen shader
    set_miss_program(current_miss_program);

	add_lights(lights);


	Logger::info << "Loading programs..." << endl;
	// Set top level geometry in acceleration structure. 
	// The default used by the ObjLoader is SBVH.

	// Set up cameras    
	Program ray_gen_program_t = context->createProgramFromPTXFile(get_path_ptx("tonemap_camera.cu"), "tonemap_camera");
	Program ray_gen_program_d = context->createProgramFromPTXFile(get_path_ptx("debug_camera.cu"), "debug_camera");

	tonemap_entry_point = add_entry_point(context, ray_gen_program_t);
	debug_entry_point = add_entry_point(context, ray_gen_program_d);

	zoomed_area = optix::make_uint4(camera_width / 2 - 5, camera_height / 2 - 5, 10, 10);
	context["zoom_window"]->setUint(zoom_debug_window);
	context["image_part_to_zoom"]->setUint(zoomed_area);
	context["debug_index"]->setUint(optix::make_uint2(0, 0));
	// Environment cameras
	

	Logger::info <<"Loading camera parameters..."<<endl;
	float max_dim = m_scene_bounding_box.extent(m_scene_bounding_box.longestAxis());
	
    //create_3d_noise(noise_frequency);

    load_camera_extrinsics(init_camera_data);

	// Set ray tracing epsilon for intersection tests
	float scene_epsilon = 1.e-4f * max_dim;
	context["scene_epsilon"]->setFloat(scene_epsilon);
	// Prepare to run 

	gui = std::make_unique<ImmediateGUI>();

    ObjMaterial params;

    params.illum = 12;
    params.ambient_tex = loadTexture(m_context, "", make_float3(0));
    params.diffuse_tex = loadTexture(m_context, "", make_float3(1, 0, 0));
    params.specular_tex = loadTexture(m_context, "", make_float3(0));
	params.name = "ketchup";

    material_ketchup = std::make_shared<MaterialHost>(context,params);

	context["show_difference_image"]->setInt(show_difference_image);
	context["merl_brdf_multiplier"]->setFloat(make_float3(1));

	context->validate();

	Logger::info << "Compiling context and creating bvhs..." << endl;

	RayGenCameraData dummy;
	dummy.eye = make_float3(0,0,0);
	dummy.U = make_float3(1,0,0);
	dummy.V = make_float3(0,1,0);
	dummy.W = make_float3(0,0,1);
	context["max_depth"]->setInt(0);
	trace(dummy);
	context["max_depth"]->setInt(ConfigParameters::get_parameter<int>("config", "max_depth", 5, "Maximum recursion depth of the raytracer"));
	reset_renderer();
    Logger::info<<"Scene initialized."<<endl;
	 //std::stringstream ss;
	 //cereal::JSONOutputArchive archive(ss);
	 //archive(*mMeshes[0]);
	 //Logger::info << ss.str() << std::endl;
	//Logger::set_logger_output(console_log);

}

void ObjScene::trace(const RayGenCameraData& s_camera_data, bool& display)
{
	display = true;
	if (mbPaused)
		return;
	context["use_heterogenous_materials"]->setInt(use_heterogenous_materials);

    mScene->get_current_camera()->update_camera(s_camera_data);

    context["tonemap_multiplier"]->setFloat(tonemap_multiplier);
	context["tonemap_exponent"]->setFloat(tonemap_exponent);

	auto total0 = currentTime();

    auto t0 = currentTime();
	if (m_camera_changed)
	{
		reset_renderer();
		m_camera_changed = false;
	}

	context["frame"]->setUint(m_frame++);

    mScene->pre_trace();
    auto t1 = currentTime();
    render_time_pre_trace = t1-t0;

    unsigned int width = mScene->get_current_camera()->get_width();
	unsigned int height = mScene->get_current_camera()->get_height();

	t0 = currentTime();
    mScene->trace();

	t1 = currentTime();
    render_time_main = t1-t0;
	// cout << "Elapsed (ray tracing): " << (time1 - time) * 1000 << endl;
	// Apply tone mapping
	t0 = currentTime();

    mScene->post_trace();

	t1 = currentTime();
    render_time_post = t1-t0;


	t0 = currentTime();
	context->launch(tonemap_entry_point, width, height);

	if (debug_mode_enabled)
	{
		context["zoom_window"]->setUint(zoom_debug_window);
		context["image_part_to_zoom"]->setUint(zoomed_area);
		context->launch(debug_entry_point, width, height);
	}
	t1 = currentTime();
    render_time_tonemap = t1-t0;

	auto total1 = currentTime();

	if (current_render_task->is_active())
	{
		if (current_render_task->is_finished())
		{
			export_raw(current_render_task->get_destination_file(), rendering_output_buffer, m_frame);
			ConfigParameters::dump_used_parameters(current_render_task->get_destination_file() + ".xml");
			current_render_task->end();
		}
        auto l = total1 - total0;

		current_render_task->update(to_milliseconds(l) / 1000.0f);
	}

	collect_image(m_frame);

	/*
    for(int i = 0; i < 200; i++)
    {
        RTbuffer * x = new RTbuffer;
        RTresult a = rtContextGetBufferFromId(m_context->get(), i, x);
        if(a == RT_SUCCESS)
        {
            RTsize w;
            m_context->getBufferFromId(i)->getSize(w);
            Logger::info << i <<":" << w << std::endl;
        }
    }*/
}

Buffer ObjScene::get_output_buffer()
{
	return returned_buffer;
}

void ObjScene::set_render_task(std::unique_ptr<RenderTask>& task)
{
	if (!current_render_task->is_active())
		current_render_task = std::move(task);
	else
		Logger::error << "Wait of end of current task before setting a new one." << endl;
}

void ObjScene::start_render_task_on_scene_ready()
{
	start_render_task_when_ready = true;
}

void ObjScene::scene_initialized()
{
	if (start_render_task_when_ready)
		current_render_task->start();
}

void ObjScene::add_override_material_file(std::string mat)
{
	override_mat = mat;
}

optix::Buffer ObjScene::createPBOOutputBuffer(const char* name, RTformat format, RTbuffertype type, unsigned width, unsigned height)
{
    Buffer buffer = context->createBuffer(type);
	buffer->setFormat(format);
    buffer->setSize(width, height);
    context[name]->setBuffer(buffer);
	return buffer;
}

void ObjScene::add_lights(vector<TriangleLight>& area_lights)
{
	Logger::info << "Adding light buffers to scene..." << endl;
    LightType::Type default_light_type = LightType::to_enum(ConfigParameters::get_parameter<string>("light", "default_light_type", LightType::to_string(LightType::DIRECTIONAL), "Type of the default light"));

	float3 light_dir = ConfigParameters::get_parameter<float3>("light","default_directional_light_direction", make_float3(0.0f, -1.0f, 0.0f), "Direction of the default directional light");
	float3 light_radiance = ConfigParameters::get_parameter<float3>("light", "default_directional_light_intensity", make_float3(5.0f), "Intensity of the default directional light");
	float3 light_pos = ConfigParameters::get_parameter<float3>("light", "default_point_light_position", make_float3(0.08f, 0.1f, 0.11f), "Position of the default point light.");
	float3 light_intensity = ConfigParameters::get_parameter<float3>("light", "default_point_light_intensity", make_float3(0.05f), "Intensity of the default point light.");
	int shadows = ConfigParameters::get_parameter<int>("light", "shadows", 1, "Use shadows in rendering.");

	std::string ptx_path_light = get_path_ptx("light_programs.cu");

    Buffer dir_light_buffer = create_buffer<SingularLightData>(context);
    Buffer area_light_buffer = create_buffer<TriangleLight>(context);

    context["light_type"]->setInt(static_cast<unsigned int>(default_light_type));
	switch (default_light_type)
	{
	case LightType::SKY:
		{
			SingularLightData light;

			static_cast<const SkyModel&>(mScene->get_miss_program()).get_directional_light(light);
            memcpy(dir_light_buffer->map(), &light, sizeof(SingularLightData));
            dir_light_buffer->unmap();
		}
		break;
	case LightType::DIRECTIONAL:
		{
            SingularLightData light = { normalize(light_dir), LightType::DIRECTIONAL, light_radiance, shadows };
            memcpy(dir_light_buffer->map(), &light, sizeof(SingularLightData));
            dir_light_buffer->unmap();
		}
		break;
	case LightType::POINT:
		{
            SingularLightData light = { light_pos, LightType::POINT, light_intensity, shadows };
            memcpy(dir_light_buffer->map(), &light, sizeof(SingularLightData));
            dir_light_buffer->unmap();
		}
		break;
	case LightType::AREA:
		{
			size_t size = area_lights.size();
			if (size == 0)
			{
				Logger::warning << "Warning: no area lights in scene. " <<
					"The only contribution will come from the ambient light (if any). " <<
					"Objects are emissive if their ambient color is not zero." << endl;

				// Dummy light to evaluate environment map.
				float3 zero = make_float3(0.0f);
				TriangleLight t = {zero, zero, zero, zero, zero, 0.0f};
				area_lights.push_back(t);
				size++;
			}

            area_light_buffer->setSize(size);

			if (size > 0)
			{
                memcpy(area_light_buffer->map(), &area_lights[0], size * sizeof(TriangleLight));
                area_light_buffer->unmap();
			}
		}
		break;
	default: break;
	}

	context["singular_lights"]->set(dir_light_buffer);
	context["area_lights"]->set(area_light_buffer);
}




bool ObjScene::export_raw(const string& raw_p, optix::Buffer out, int frames)
{
	std::string raw_path = raw_p;
	// export render data
    if (raw_path.length() == 0)
    {
        Logger::error << "Invalid raw file specified" << raw_path << endl;
        return false;
    }

	if (raw_path.length() <= 4 || raw_path.substr(raw_path.length() - 4).compare(".raw") != 0)
	{
        raw_path += ".raw";
	}

	RTsize w, h;
	out->getSize(w, h);
	std::string txt_file = raw_path.substr(0, raw_path.length() - 4) + ".txt";
	ofstream ofs_data(txt_file);
	if (ofs_data.bad())
	{
		Logger::error <<  "Unable to open file " << txt_file << endl;
		return false;
	}
	ofs_data << frames << endl << w << " " << h << endl;
	ofs_data << 1.0 << " " << 1.0f << " " << 1.0f;
	ofs_data.close();

	RTsize size_buffer = w * h * 4;
	float* mapped = new float[size_buffer];
	memcpy(mapped, out->map(), size_buffer * sizeof(float));
	out->unmap();
	ofstream ofs_image;
	ofs_image.open(raw_path, ios::binary);
	if (ofs_image.bad())
	{
		Logger::error <<"Error in exporting file"<<endl;
		return false;
	}

	RTsize size_image = w * h * 3;
	float* converted = new float[size_image];
	float average = 0.0f;
	float maxi = -INFINITY;
	for (int i = 0; i < size_image / 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			if (!isfinite(mapped[i * 4 + j]))
			{
			}
			converted[i * 3 + j] = mapped[i * 4 + j];
			average += mapped[i * 4 + j];
			maxi = max(maxi, mapped[i * 4 + j]);
		}
	}
	average /= size_image * 3;
	delete[] mapped;
	ofs_image.write(reinterpret_cast<const char*>(converted), size_image * sizeof(float));
	ofs_image.close();
	delete[] converted;
	Logger::info <<"Exported buffer to " << raw_path << " (avg: " << to_string(average) << ", max: "<< to_string(maxi) <<")" <<endl;

	return true;
}

void ObjScene::do_resize(unsigned int , unsigned int )
{
}

void ObjScene::resize(unsigned int width, unsigned int height)
{
	do_resize(width, height);
}

bool ObjScene::mouse_moving(int x, int y)
{
	return gui->mouseMoving(x, y);
}

void ObjScene::post_draw_callback()
{
	if (!gui->isVisible())
		return;
	gui->start_window("Ray tracing demo", 20, 20, 500, 600);
	if (draw_gui())
	{
		reset_renderer();
	}
	gui->end_window();
}

void ObjScene::set_debug_pixel(unsigned int x, unsigned int y)
{
	Logger::info <<"Setting debug pixel to " << to_string(x) << " << " << to_string(y) <<endl;
	context->setPrintLaunchIndex(x, y);
	context["debug_index"]->setUint(x, y);
}

bool ObjScene::mouse_pressed(int x, int y, int button, int action, int mods)
{
	y = mScene->get_current_camera()->get_height() - y;
	if (button == GLFW_MOUSE_BUTTON_RIGHT && debug_mode_enabled && x > 0 && y > 0)
	{
		set_debug_pixel(static_cast<unsigned int>(x), static_cast<unsigned int>(y));
		zoomed_area = optix::make_uint4(x - zoomed_area.z / 2, y - zoomed_area.w / 2, zoomed_area.z, zoomed_area.w);
		return true;
	}
	return gui->mousePressed(x, y, button, action, mods);
}

void ObjScene::set_miss_program(BackgroundType::EnumType program)
{
    std::unique_ptr<MissProgram> miss_program = nullptr;

	switch (program)
	{
	case BackgroundType::ENVIRONMENT_MAP:
	{
        string env_map_name = ConfigParameters::get_parameter<string>("light", "environment_map", "pisa.hdr", "Environment map file");
        miss_program = std::make_unique<EnvironmentMap>(env_map_name);
	}
    break;
	case BackgroundType::SKY_MODEL:
    {
        miss_program = std::make_unique<SkyModel>(make_float3(0, 1, 0), make_float3(0, 0, 1));
	}
    break;
    case BackgroundType::CONSTANT_BACKGROUND:
	default:
        float3 color = ConfigParameters::get_parameter<float3>("light", "background_constant_color", make_float3(0.5), "Environment map file");
        miss_program = std::make_unique<ConstantBackground>(color);
        break;
    }
    mScene->set_miss_program(std::move(miss_program));
}

void ObjScene::set_rendering_method(RenderingMethodType::EnumType t)
{
    std::unique_ptr<RenderingMethod> m;
	switch (t)
	{
	case RenderingMethodType::RECURSIVE_RAY_TRACING:
		m = std::make_unique<SimpleTracing>(context);
		break;
	case RenderingMethodType::AMBIENT_OCCLUSION:
		m = std::make_unique<AmbientOcclusion>(context);
		break;
	case RenderingMethodType::PATH_TRACING:
		m = std::make_unique<PathTracing>(context);
		break;
	default:
		Logger::error<<"The selected rendering method is not valid or no longer supported."<< endl;
		break;
	}
    mScene->set_method(std::move(m));
}


void ObjScene::setDebugEnabled(bool var)
{
	debug_mode_enabled = var;
	if (var)
	{
		context->setPrintEnabled(true);
		context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
	}
	else
	{
		context->setPrintEnabled(false);
		context->setExceptionEnabled(RT_EXCEPTION_ALL, false);
	}
}


void ObjScene::load_camera_extrinsics(InitialCameraData & camera_data)
{
    auto camera_type = PinholeCameraType::to_enum(ConfigParameters::get_parameter<string>("camera", "camera_definition_type", PinholeCameraType::to_string(PinholeCameraType::EYE_LOOKAT_UP_VECTORS), "Type of the camera."));  

	float max_dim = m_scene_bounding_box.extent(m_scene_bounding_box.longestAxis());
	float3 eye = m_scene_bounding_box.center();
	eye.z += 3 * max_dim;

	bool use_auto_camera = ConfigParameters::get_parameter<bool>("camera", "use_auto_camera", false, "Use a automatic placed camera or use the current data.");

    optix::Matrix3x3 camera_matrix = optix::Matrix3x3::identity();

	float vfov = ConfigParameters::get_parameter<float>("camera", "camera_fov", 53, "The camera FOVs (h|v)");

	float ratio = mScene->get_current_camera()->get_width() / (float)mScene->get_current_camera()->get_height();
	float hfov = rad2deg(2.0f*atanf(ratio*tanf(deg2rad(0.5f*(vfov)))));

    if (use_auto_camera)
	{
		camera_data = InitialCameraData(eye, // eye
			m_scene_bounding_box.center(), // lookat
			make_float3(0.0f, 1.0f, 0.0f), // up
			hfov, vfov);
	}
	else
	{
		eye = ConfigParameters::get_parameter<float3>("camera", "camera_position", make_float3(1, 0, 0), "The camera initial position");
		float3 lookat = ConfigParameters::get_parameter<float3>("camera", "camera_lookat_point", make_float3(0, 0, 0), "The camera initial lookat point");
		float3 up = ConfigParameters::get_parameter<float3>("camera", "camera_up", make_float3(0, 1, 0), "The camera initial up");
		camera_data = InitialCameraData(eye, lookat, up, hfov, vfov);
	}

	if (camera_type == PinholeCameraType::INVERSE_CAMERA_MATRIX)
	{
		camera_matrix = ConfigParameters::get_parameter<optix::Matrix3x3>("camera", "inv_camera_matrix", optix::Matrix3x3::identity(), "The camera inverse calibration matrix K^-1 * R^-1");
	}

	reset_renderer();
}

