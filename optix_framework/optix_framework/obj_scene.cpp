// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <fstream>
#include "obj_loader.h"
#include "singular_light.h"
#include "obj_scene.h"
#include "simple_tracing.h"
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
#include "environment_map_background.h"
#include "constant_background.h"
#include "PerlinNoise.h"
#include <sampled_bssrdf.h>
#include "render_task.h"
#include "volume_path_tracer.h"
#include "GLFW/glfw3.h"
#include "cputimer.h"
#include "sky_model.h"
#include "bssrdf_visualizer.h"
#include <algorithm>

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


ObjScene::ObjScene(const std::vector<std::string>& obj_filenames)
	: context(m_context),
	filenames(obj_filenames), m_frame(0u)
{
	current_render_task = make_unique<RenderTaskFrames>(1000, "res.raw", false);
}

ObjScene::ObjScene()
	: context(m_context),
     filenames(1, "test.obj"),
	m_frame(0u)
{
	current_render_task = make_unique<RenderTaskFrames>(1000, "res.raw", false);
}

inline ObjScene::~ObjScene()
{
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
	static bool debug = parameters.debug_enabled;

	if(ImmediateGUIDraw::Button("Dump buffers"))
	{
		for(int i = 0; i < 250; i++)
		{
			RTbuffer buffer;
			auto res = rtContextGetBufferFromId( m_context->get(), i, &buffer );
			if(res == RT_SUCCESS)
			{

				optix::Buffer b = Buffer::take(buffer);
				RTsize d = b->getDimensionality();
				RTsize * dims = new RTsize[d];
				b->getSize(d, &dims[0]);
				void** device_pointer;
				auto r = rtBufferGetDevicePointer (buffer, 0, device_pointer);
				if(r == RT_SUCCESS)
					printf("Buffer %d, size %d (w %d). ptr %p\n", i, b->getElementSize(), dims[0], *device_pointer);
				else
					printf("Buffer %d, size %d (w %d).\n", i, b->getElementSize(), dims[0]);
			}
		}
	}

	if(ImmediateGUIDraw::Button("Serialize"))
	{
        std::string path;
        if(Dialogs::saveFileDialog(path))
        {
            serialize_scene(path);
        }
        changed = true;
	}

    if(ImmediateGUIDraw::Button("Load scene..."))
    {
        std::string path;
        if(Dialogs::openFileDialog(path))
        {
            load_scene(path);
        }
        changed = true;
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
            load_default_camera();
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
		changed |= ImmediateGUIDraw::SliderFloat("Multiplier##TonemapMultiplier", &tonemap_parameters.multiplier, 0.0f, 2.0f, "%.3f", 1.0f);
		changed |= ImmediateGUIDraw::SliderFloat("Exponent##TonemapExponent", &tonemap_parameters.exponent, 0.5f, 3.5f, "%.3f", 1.0f);
		if (ImmediateGUIDraw::Button("Reset##TonemapExponentMultiplierReset"))
		{
			changed = true;
            tonemap_parameters = TonemapParameters();
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

    changed = mScene->on_draw();

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
			start_task_render_time = currentTime();
			current_render_task->start();
		}

		if (current_render_task->is_active() && ImmediateGUIDraw::Button("End task"))
		{
			current_render_task->end();
		}
	}

	return changed;
}

void ObjScene::serialize_scene(const std::string &dest)
{
    Logger::info << "Serializing..." << std::endl;
    {
        cereal::XMLOutputArchiveOptix output_archive(dest, true);
        output_archive(mScene);
    }
}


void ObjScene::create_3d_noise(float frequency)
{
    static Texture tex(context);
    tex.set_size(256,256,256);

    static PerlinNoise p(1337);
    // Create buffer with single texel set to default_color
    float* buffer_data = new float[256*256*256];

    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j++)
            for (int k = 0; k < 256; k++)
            {
                int idx = 256 * 256 * i + 256 * j + k;
                buffer_data[idx] = (float)p.noise(i / (256.0f) * frequency, j / (256.0f) * frequency, k / (256.0f) * frequency);
            }
    tex.set_data(buffer_data, 256*256*256*sizeof(float));
    context["noise_tex"]->setInt(tex.get_id());
}



void ObjScene::initialize_scene(GLFWwindow *)
{
	Logger::info << "Initializing scene." << endl;

	context->setPrintBufferSize(parameters.print_buffer_size);
    context->setPrintLaunchIndex(parameters.print_index.x,parameters.print_index.y);
    context->setRayTypeCount(RayType::count());
    context->setStackSize(parameters.stack_size);
    context["bad_color"]->setFloat(parameters.exception_color);

	setDebugEnabled(parameters.debug_enabled);

    std::string mpml_file = Folders::data_folder + "/mpml/media.mpml";

    if(exists(mpml_file.c_str()))
    	MaterialLibrary::load(Folders::data_folder.c_str());

    ScatteringMaterial::initializeDefaultMaterials();
    load_parameters("ray_tracing_parameters.xml");

    ShaderFactory::init(context);
    ShaderInfo info = ShaderInfo(12, "volume_shader.cu", "Volume path tracer");
    ShaderFactory::add_shader(std::make_unique<VolumePathTracer>(info));

    ShaderInfo info3 = ShaderInfo(13, "volume_shader_heterogenous.cu", "Volume path tracer (het.)");
    ShaderFactory::add_shader(std::make_unique<VolumePathTracer>(info3));

    ShaderInfo info2 = ShaderInfo(14, "subsurface_scattering_sampled_default.cu", "Sampled BSSRDF");
    ShaderFactory::add_shader(std::make_unique<SampledBSSRDF>(info2));

    for (auto& kv : MaterialLibrary::media)
    {
        available_media.push_back(&kv.second);
    }

    auto res = std::find_if(filenames.begin(), filenames.end(), [](const std::string & s) { return s.substr(s.size()-4, s.size()) == ".xml";});
    if(res != filenames.end())
    {
        // Scene found!
		load_scene(*res);
        filenames.erase(res);
    }
    else
    {
        mScene = std::make_unique<Scene>(context);

        CameraParameters params;
        params.width = 512;
        params.height = 512;
        params.downsampling = 1;
        params.vfov = 53.0f;
        float ratio = params.width / (float)params.height;
        params.hfov = rad2deg(2.0f*atanf(ratio*tanf(deg2rad(0.5f*(params.vfov)))));
        params.rendering_rect = optix::make_int4(-1);
        auto id = mScene->add_camera(std::make_unique<Camera>(context, params));
        mScene->set_current_camera(id);
        RenderingMethodType::EnumType t = RenderingMethodType::PATH_TRACING;
        set_rendering_method(t);
		mScene->set_miss_program(std::make_unique<ConstantBackground>(optix::make_float3(0.5f)));

        // Camera must be automatic in this case
        parameters.use_auto_camera = true;
    }


    std::shared_ptr<Camera> camera = mScene->get_current_camera();
    unsigned int camera_width = camera->get_width();
    unsigned int camera_height = camera->get_height();

	// Tone mapping pass
	rendering_output_buffer = createPBOOutputBuffer("output_buffer", RT_FORMAT_FLOAT4, RT_BUFFER_INPUT_OUTPUT, camera->get_width(), camera->get_height());
	tonemap_output_buffer = createPBOOutputBuffer("tonemap_output_buffer", RT_FORMAT_UNSIGNED_BYTE4, RT_BUFFER_INPUT_OUTPUT, camera->get_width(), camera->get_height());
	debug_output_buffer = createPBOOutputBuffer("debug_output_buffer", RT_FORMAT_UNSIGNED_BYTE4, RT_BUFFER_OUTPUT, camera->get_width(), camera->get_height());
	returned_buffer = tonemap_output_buffer;

	// Create group for scene objects and float acceleration structure

	// We need the scene bounding box for placing the camera
	Aabb bbox(make_float3(-1,-1,-1), make_float3(1,1,1));

	// Load geometry from OBJ files into the group of scene objects
	Logger::info << "Loading obj files..." << endl;

	m_scene_bounding_box = bbox;
	for (unsigned int i = 0; i < filenames.size(); ++i)
	{
		// Load OBJ scene
		Logger::info <<"Loading obj " << filenames[i]  << "..." <<endl;
		ObjLoader loader((Folders::data_folder + filenames[i]).c_str(), context);
        std::vector<std::unique_ptr<Object>>& v = loader.load(optix::Matrix4x4::identity());
		for (int j = 0; j < v.size(); j++)
		{
             mScene->add_object(std::move(v[j]));
		}

	    m_scene_bounding_box.include(loader.getSceneBBox());
		// Set material shaders


	}
	
	context["importance_sample_area_lights"]->setUint(static_cast<unsigned int>(mImportanceSampleAreaLights));

    load_default_camera();

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


	// Set ray tracing epsilon for intersection tests
	context["scene_epsilon"]->setFloat(parameters.scene_epsilon_fraction * max_dim);
	// Prepare to run 

	if(GLFWDisplay::isDisplayAvailable())
		gui = std::make_unique<ImmediateGUI>();

	context["show_difference_image"]->setInt(show_difference_image);

	context->validate();

	Logger::info << "Compiling context and creating bvhs..." << endl;

	context["max_depth"]->setInt(0);
	trace();
	context["max_depth"]->setInt(parameters.max_depth);
	reset_renderer();
    Logger::info<<"Scene initialized."<<endl;

	if(!exists("ray_tracing_parameters.xml"))
	    save_parameters("ray_tracing_parameters.xml");


}

void ObjScene::trace()
{
	if (mbPaused)
		return;

    context["tonemap_multiplier"]->setFloat(tonemap_parameters.multiplier);
	context["tonemap_exponent"]->setFloat(tonemap_parameters.exponent);

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

	if (parameters.debug_enabled)
	{
		context["zoom_window"]->setUint(zoom_debug_window);
		context["image_part_to_zoom"]->setUint(zoomed_area);
		context->launch(debug_entry_point, width, height);
	}

	t1 = currentTime();
    render_time_tonemap = t1-t0;

    std::chrono::time_point<std::chrono::high_resolution_clock> total1 = currentTime();

	if (current_render_task->is_active())
	{
		if (current_render_task->is_finished())
		{
			export_raw(current_render_task->get_destination_file(), rendering_output_buffer, m_frame);
			
			current_render_task->end();
		}
        std::chrono::duration<double> l = total1 - start_task_render_time;
		current_render_task->update_absolute(l.count());
	}



	collect_image(m_frame);
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
	{
		current_render_task->start();
		start_task_render_time = currentTime();
	}
}

optix::Buffer ObjScene::createPBOOutputBuffer(const char* name, RTformat format, RTbuffertype type, unsigned width, unsigned height)
{
    Buffer buffer = context->createBuffer(type);
	buffer->setFormat(format);
    buffer->setSize(width, height);
    context[name]->setBuffer(buffer);
	return buffer;
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
	if (gui == nullptr || !gui->isVisible())
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
	if (button == GLFW_MOUSE_BUTTON_RIGHT && parameters.debug_enabled && x > 0 && y > 0)
	{
		set_debug_pixel(static_cast<unsigned int>(x), static_cast<unsigned int>(y));
		zoomed_area = optix::make_uint4(x - zoomed_area.z / 2, y - zoomed_area.w / 2, zoomed_area.z, zoomed_area.w);
		return true;
	}
	return gui->mousePressed(x, y, button, action, mods);
}

void ObjScene::set_rendering_method(RenderingMethodType::EnumType t)
{
    std::unique_ptr<RenderingMethod> m;
	switch (t)
	{
	case RenderingMethodType::RECURSIVE_RAY_TRACING:
		m = std::make_unique<SimpleTracing>();
		break;
	case RenderingMethodType::AMBIENT_OCCLUSION:
		m = std::make_unique<AmbientOcclusion>();
		break;
	case RenderingMethodType::PATH_TRACING:
		m = std::make_unique<PathTracing>();
		break;
	default:
		Logger::error<<"The selected rendering method is not valid or no longer supported."<< endl;
		break;
	}
    mScene->set_method(std::move(m));
}


void ObjScene::setDebugEnabled(bool var)
{
	parameters.debug_enabled = var;
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


void ObjScene::load_default_camera()
{
	float max_dim = m_scene_bounding_box.extent(m_scene_bounding_box.longestAxis());
	float3 eye = m_scene_bounding_box.center();
	eye.z += 3 * max_dim;

    if (parameters.use_auto_camera)
	{
        mScene->get_current_camera()->setEyeLookatUp(eye, m_scene_bounding_box.center(), make_float3(0.0f, 1.0f, 0.0f));
	}
    reset_renderer();
}

Camera *ObjScene::get_camera()
{
    return mScene->get_current_camera().get();
}


template<class Archive>
void serialize(Archive& ar, RayTracerParameters & in)
{
    ar(
            cereal::make_nvp("print_buffer_size", in.print_buffer_size),
            cereal::make_nvp("print_index", in.print_index),
            cereal::make_nvp("debug_enabled", in.debug_enabled),
            cereal::make_nvp("stack_size", in.stack_size),
            cereal::make_nvp("exception_color", in.exception_color),
            cereal::make_nvp("max_depth", in.max_depth),
            cereal::make_nvp("scene_epsilon_fraction", in.scene_epsilon_fraction),
            cereal::make_nvp("use_auto_camera", in.use_auto_camera)
    );
}

template<class Archive>
void serialize(Archive& ar, TonemapParameters & in)
{
    ar(
            cereal::make_nvp("exponent", in.exponent),
            cereal::make_nvp("multiplier", in.multiplier)
    );
}

void ObjScene::load_parameters(const std::string &config_file)
{
    if(!exists(config_file.c_str()))
        return;
    cereal::XMLInputArchiveOptix archive(context, config_file);
    archive(cereal::make_nvp("data_folder",Folders::data_folder));
    archive(cereal::make_nvp("ptx_folder",Folders::ptx_path));
    archive(cereal::make_nvp("renderer_parameters",parameters));
    archive(cereal::make_nvp("tonemap_parameters",tonemap_parameters));
}

void ObjScene::save_parameters(const std::string &config_file)
{
    cereal::XMLOutputArchiveOptix archive(config_file);
    archive(cereal::make_nvp("data_folder",Folders::data_folder));
    archive(cereal::make_nvp("ptx_folder",Folders::ptx_path));
    archive(cereal::make_nvp("renderer_parameters",parameters));
    archive(cereal::make_nvp("tonemap_parameters",tonemap_parameters));
}

void ObjScene::load_scene(const std::string &str)
{
    Logger::info << "Loading..." << std::endl;
    {
        cereal::XMLInputArchiveOptix input_archive(m_context, str, true);
        input_archive(mNewScene);
    }
    mScene = std::move(mNewScene);
}
