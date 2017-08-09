// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#define ECO3D_SCENE
#include <iostream>
#include <fstream>
#include <string>
#include <sutil.h>
#include "obj_loader.h"
#include "directional_light.h"
#include "point_light.h"
#include "obj_scene.h"
#include "folders.h"
#include "simple_tracing.h"
#include "parameter_parser.h"
#include "HDRLoader.h"
#include "material_library.h"
#include "ambient_occlusion.h"
#include "path_tracing.h"
#include "logger.h"
#include "procedural_loader.h"
#include "sphere.h"
#include "dialogs.h"

#include <sutil/ImageLoader.h>
#include "presampled_surface_bssrdf.h"
#include "apple_juice.h"
#include "GEL/GLGraphics/SOIL.h"
#include "GLUTDisplay.h"
#include <optprops/glass.h>
#include <GEL/GL/glut.h>
#include "optical_helper.h"
#include "CGLA/Mat3x3f.h"
#include "aisceneloader.h"
#include "shader_factory.h"
#include "scattering_material.h"

using namespace std;
using namespace optix;
using namespace CGLA;

void ObjScene::add_result_image(const string& image_file)
{
	comparison_image = loadTexture(context->getContext(), image_file, optix::make_float3(0, 1, 0));
	gui->setVisible("Weight", true);
	context["comparison_texture"]->setInt(comparison_image->getId());
}


void ObjScene::execute_on_scene_elements(function<void(Mesh&)> operation) 
{
    for (Mesh & m : mMeshes)
    {
        operation(m);
    }
}

void ObjScene::start_simulation()
{
	collect_images = true;
	init_simulation(m_simulation_parameters);
	reset_renderer();
	gui->setReadOnly("Start Simulation");
	gui->setReadOnly("Starting");
	gui->setReadOnly("Ending");
	gui->setReadOnly("Step");
	gui->setReadOnly("Samples");
}

void ObjScene::end_simulation()
{
	collect_images = false;
	gui->setReadWrite("Start Simulation");
	gui->setReadWrite("Starting");
	gui->setReadWrite("Ending");
	gui->setReadWrite("Step");
	gui->setReadWrite("Samples");
}

void ObjScene::collect_image(unsigned int frame)
{
	if (mAutoMode)
	{
		if (frame == mFrames)
		{
			std::string name = mOutputFile;
			export_raw(name);
			exit(2);
		}
		else
		{
			return;
		}
	}

	if (!collect_images) return;

	if (frame == m_simulation_parameters.samples)
	{
		std::string name = std::string("rendering_") + get_name() + ".raw";
		export_raw(name);

		update_simulation(m_simulation_parameters);
		reset_renderer();
		if (m_simulation_parameters.status == SimulationParameters::FINISHED)
		{
			end_simulation();
		}
	}
}

void ObjScene::reset_renderer()
{
	m_frame = 0;
}

void ObjScene::setAutoMode()
{
	mAutoMode = true;
}

void ObjScene::setOutputFile(const string& cs)
{
	mOutputFile = cs;
}

void ObjScene::setFrames(int frames)
{
	mFrames = frames;
}

bool ObjScene::keyPressed(unsigned char key, int x, int y)
{
	if (mAutoMode)
		return false;
	if (gui->keyPressed(key, x, y) || key >= 48 && key <= 57) // numbers avoided
	{
		reset_renderer();
		return true;
	}
	switch (key)
	{
	case 't':
		use_tonemap = !use_tonemap;
		return true;
	case 'e':
		{
			std::string res = std::string("result_optix.raw");
			return export_raw(res);
		}
	case 'p':
		//current_scene_type = Scene::NextEnumItem(current_scene_type);
		if (current_scene_type == Scene::NotValidEnumItem)
			current_scene_type = Scene::OPTIX_ONLY;
		reset_renderer();
		/*	  Buffer out = context[OUTPUT_BUFFER]->getBuffer();
			  out->destroy();
			  context[OUTPUT_BUFFER]->setBuffer(createPBOOutputBuffer(RT_FORMAT_FLOAT4, window_width, window_height));
			  cout << ((use_optix) ? "Ray tracing" : "Rasterization") << endl;*/
		return true;
	case 'n':
		{
			start_simulation();
			return true;
		}
		break;
	case 'g':
		{
			gui->toggleVisibility();
			return true;
		}
		break;
	default: return false;
	}
}


void ObjScene::initUI()
{
	Logger::info << "Initializing UI..." << endl;
	gui->addIntVariable("Frames", reinterpret_cast<int*>(&m_frame), "Settings");
	gui->setReadOnly("Frames");
	gui->addCheckBoxCallBack("Debug Mode", setDebugMode, getDebugMode, this, "Settings");
	gui->addIntVariableCallBack("Depth", setRTDepth, getRTDepth, this, "Settings", 0, 100000, 1);

	const char * comp_group = "Comparison";
	gui->addButton("Load result image", loadImage, this, comp_group);
	gui->addFloatVariable("Weight", &comparison_image_weight, comp_group, 0.0f, 1.0f, 0.01f);
	gui->addCheckBox("Difference image", &show_difference_image, comp_group);
	gui->setVisible("Weight", false);

	const char* tmg = "Tone mapping";
	gui->addCheckBox("Tonemap", &use_tonemap, tmg);
	gui->addFloatVariable("Multiplier", &tonemap_multiplier, tmg, 0.0f, 1.0f, 0.05f);
	gui->addFloatVariable("Exponent", &tonemap_exponent, tmg, 0.5f, 3.5f, 0.05f);

	const char* glass_group = "Glass";
	gui->addFloatVariableCallBack("Index of refraction", setIor, getIor, this, glass_group);
	gui->addFloatVariableCallBack("Absorption - R", setAbsorptionColorR, getAbsorptionColorR, this, glass_group);
	gui->addFloatVariableCallBack("Absorption - G", setAbsorptionColorG, getAbsorptionColorG, this, glass_group);
	gui->addFloatVariableCallBack("Absorption - B", setAbsorptionColorB, getAbsorptionColorB, this, glass_group);
	gui->addFloatVariableCallBack("Absorption inv. multiplier", setAbsorptionInverseMultiplier, getAbsorptionInverseMultiplier, this, glass_group);

    // Read only char
	//gui->addFloatVariable("Absorption - R", &calc_absorption[0], glass_group);
	//gui->setReadOnly("Absorption - R");
	//gui->addFloatVariable("Absorption - G", &calc_absorption[1], glass_group);
	//gui->setReadOnly("Absorption - G");
	//gui->addFloatVariable("Absorption - B", &calc_absorption[2], glass_group);
	//gui->setReadOnly("Absorption - B");

	vector<GuiDropdownElement> elems;
	int count = 0;
	for (const auto& kv : available_media)
	{
		GuiDropdownElement e = { count++, kv->name.c_str() };
		elems.push_back(e);
	}
	gui->addDropdownMenuCallback("Medium (glass)", elems, setMedium, getMedium, this, glass_group);

	const char* env_map_correction_group = "Environment map corrections";
	gui->addFloatVariable("Lightmap multiplier - R", &lightmap_multiplier.x, env_map_correction_group);
	gui->addFloatVariable("Lightmap multiplier - G", &lightmap_multiplier.y, env_map_correction_group);
	gui->addFloatVariable("Lightmap multiplier - B", &lightmap_multiplier.z, env_map_correction_group);
	gui->addFloatVariableCallBack("Delta X", setDeltaX, getDeltaX, this, env_map_correction_group, -180.0, 180.0f, .010f);
	gui->addFloatVariableCallBack("Delta Y", setDeltaY, getDeltaY, this, env_map_correction_group, -180.0, 180.0f, .010f);
	gui->addFloatVariableCallBack("Delta Z", setDeltaZ, getDeltaZ, this, env_map_correction_group, -180.0, 180.0f, .010f);

	gui->addButton("Reset Camera", resetCameraCallback, this, "Settings");
	gui->addButton("Save RAW File", saveRawCallback, this, "Settings");



    execute_on_scene_elements([=](Mesh & m)
    {
        if (m.mMaterialData.illum == 17 || m.mMaterialData.illum == 12)
        {
            m.mMaterialData.scattering_material->set_into_gui(gui);
        }
    });

	// Simulation UI
	//const char * simulation_group = "Simulation";
	//gui->addFloatVariable("Starting", &m_simulation_parameters.start, simulation_group, -FLT_MAX, FLT_MAX);
	//gui->addFloatVariable("Ending", &m_simulation_parameters.end, simulation_group, -FLT_MAX, FLT_MAX);
	//gui->addFloatVariable("Step", &m_simulation_parameters.step, simulation_group);
	//gui->addIntVariable("Samples", &m_simulation_parameters.samples, simulation_group);
	//vector<GuiDropdownElement> elems_sim = {
	//		{ SimulationParameters::SimulationElement::IOR, "Index of refraction" },
	//		{ SimulationParameters::SimulationElement::ANGLE_X, "X angle" },
	//		{ SimulationParameters::SimulationElement::ANGLE_Y, "Y angle" },
	//		{ SimulationParameters::SimulationElement::ANGLE_Z, "Z angle" },
	//		{ SimulationParameters::SimulationElement::ALL_ANGLES, "All angles" },
	//		{ SimulationParameters::SimulationElement::LIGHTMAP_MULTIPLIER, "Lightmap multiplier" },
	//		{ SimulationParameters::SimulationElement::ABSORPTION_R, "Absorption, R" },
	//		{ SimulationParameters::SimulationElement::ABSORPTION_G, "Absorption, G" },
	//		{ SimulationParameters::SimulationElement::ABSORPTION_B, "Absorption, B" },
	//		{ SimulationParameters::SimulationElement::ABSORPTION_M, "Absorption, M" }
	//};
	//gui->addDropdownMenu("Element to simulate", elems_sim, reinterpret_cast<int*>(&m_simulation_parameters.parameter_to_simulate), simulation_group);
	//gui->addButton("Start Simulation", startSimulationCallback, this, simulation_group);
	//gui->addButton("End Simulation", endSimulationCallback, this, simulation_group);

	const char * limit_rendering_group = "Rendering Bounds";
	gui->addIntVariable("X", (int*)&camera->data.render_bounds.x, limit_rendering_group, 0, camera->data.camera_size.x);
	gui->addIntVariable("Y", (int*)&camera->data.render_bounds.y, limit_rendering_group, 0, camera->data.camera_size.y);
	gui->addIntVariable("W", (int*)&camera->data.render_bounds.z, limit_rendering_group, 0, camera->data.camera_size.x);
	gui->addIntVariable("H", (int*)&camera->data.render_bounds.w, limit_rendering_group, 0, camera->data.camera_size.y);
}

void ObjScene::initScene(InitialCameraData& init_camera_data)
{
	Logger::info << "Initializing scene." << endl;
	context->setPrintBufferSize(200);
	setDebugEnabled(false);
	context->setPrintLaunchIndex(0, 0);
	ParameterParser::init(config_file);
	Folders::init();
	MaterialLibrary::load(Folders::mpml_file.c_str());


    context->setEntryPointCount(as_integer(CameraType::COUNT));
    ShaderFactory::init(context);
    for (auto& kv : MaterialLibrary::media)
	{
		available_media.push_back(&kv.second);
	}
	sky_model.init();

	

	float3 ambient_light_color_p = ParameterParser::get_parameter<float3>("light", "ambient_light_color", make_float3(0.0f), "The ambient light color");

	int camera_width = ParameterParser::get_parameter<int>("camera","window_width", 512, "The width of the window");
	int camera_height = ParameterParser::get_parameter<int>("camera", "window_height", 512, "The height of the window");
	int downsampling = ParameterParser::get_parameter<int>("camera", "camera_downsampling", 1, "");

	camera = new Camera(camera_width, camera_height, downsampling, custom_rr);



	Logger::info << "Rendering rectangle: " << camera->data.rendering_rectangle.x << " " << camera->data.rendering_rectangle.y << " " <<
		camera->data.rendering_rectangle.z << " " <<
		camera->data.rendering_rectangle.w << " Camera: " << camera_width << " " << camera_height << endl;

	default_miss = BackgroundType::String2Enum(ParameterParser::get_parameter<string>("config", "default_miss_type", BackgroundType::Enum2String(BackgroundType::CONSTANT_BACKGROUND), "Default miss program."));


	context["ambient_light_color"]->setFloat(ambient_light_color_p);
	// Setup context
	context->setRayTypeCount(3);
	context->setStackSize(ParameterParser::get_parameter<int>("config", "stack_size", 2000, "Allocated stack size for context"));

	context["radiance_ray_type"]->setUint(RAY_TYPE_RADIANCE);
	context["shadow_ray_type"]->setUint(RAY_TYPE_SHADOW);
	context["dummy_ray_type"]->setUint(RAY_TYPE_DUMMY);

	context["max_depth"]->setInt(ParameterParser::get_parameter<int>("config", "max_depth", 5, "Maximum recursion depth of the raytracer"));
	context[OUTPUT_BUFFER]->set(createPBOOutputBuffer(OUTPUT_BUFFER, RT_FORMAT_FLOAT4, camera->get_width(), camera->get_height()));

	// Constant colors
	context["bad_color"]->setFloat(0.0f, 1.0f, 0.0f);
	context["bg_color"]->setFloat(ambient_light_color_p);
	
	bool use_abs = ParameterParser::get_parameter<bool>("config", "use_absorption", true, "Use absorption in rendering.");
	Logger::debug << "Absorption is " << (use_abs ? "ON" : "OFF") << endl;

	// Tone mapping pass
	context["tonemap_output_buffer"]->set(createPBOOutputBuffer("tonemap_output_buffer", RT_FORMAT_FLOAT4, camera->get_width(), camera->get_height()));

	// Create group for scene objects and add acceleration structure
	scene = context->createGroup();
	scene->setChildCount(static_cast<unsigned int>(filenames.size()));
	Acceleration acceleration = context->createAcceleration("Trbvh");
	scene->setAcceleration(acceleration);
	acceleration->markDirty();

	// We need the scene bounding box for placing the camera
	Aabb bbox;

	
	const string ptx_path_def = get_path_ptx("pinhole_camera.cu");
	Program empty = context->createProgramFromPTXFile(ptx_path_def, "empty");
	for (int i = 0; i < as_integer(CameraType::COUNT); i++)
		context->setRayGenerationProgram(i, empty);

    RenderingMethodType::EnumType t = RenderingMethodType::String2Enum(ParameterParser::get_parameter<string>("config", "rendering_type", RenderingMethodType::Enum2String(RenderingMethodType::RECURSIVE_RAY_TRACING), "Rendering method"));
	set_rendering_method(t);

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
		GeometryGroup geometry_group = context->createGeometryGroup();
		ObjLoader* loader = new ObjLoader((Folders::data_folder + filenames[i]).c_str(), context, geometry_group);
        vector<Mesh> v = loader->load(get_object_transform(filenames[i]));
		mMeshes.insert(mMeshes.end(), v.begin(), v.end());
	    m_scene_bounding_box.include(loader->getSceneBBox());
		loader->getAreaLights(lights);
        

		delete loader;
		// Set material shaders

		// Add geometry group to the group of scene objects
		scene->setChild(i, geometry_group);
	}

    execute_on_scene_elements([=](Mesh & m)
    {
        m.load_shader(m, t);
    });

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

	scene->setChildCount(static_cast<unsigned int>(filenames.size() + procedural.size()));
	for (int i = 0; i < procedural.size(); i++)
	{
		GeometryGroup geometry_group = context->createGeometryGroup();
		ProceduralMesh* mesh = procedural[i];
		if (mesh != nullptr)
		{
			ObjLoader* loader = new ProceduralLoader(mesh, context, geometry_group);
			loader->load();
			m_scene_bounding_box.include(loader->getSceneBBox());
			loader->getAreaLights(lights);
			delete loader;

			// Set material shaders
			for (unsigned int j = 0; j < geometry_group->getChildCount(); ++j)
			{
//				GeometryInstance gi = geometry_group->getChild(j);
//				addMERLBRDFtoGeometry(gi, use_merl_brdf);

                // FIXME
				//method->init(gi);
			}

			// Add geometry group to the group of scene objects
			scene->setChild(static_cast<unsigned int>(filenames.size()) + i, geometry_group);
		}
	}

	// Add light sources depending on chosen shader

	add_lights(lights);


	Logger::info << "Loading programs..." << endl;
	// Set top level geometry in acceleration structure. 
	// The default used by the ObjLoader is SBVH.
	context["top_object"]->set(scene);
	context["top_shadower"]->set(scene);

	// Set up camera
	auto camera_type = PinholeCameraDefinitionType::String2Enum(ParameterParser::get_parameter<string>("camera", "camera_definition_type", PinholeCameraDefinitionType::Enum2String(PinholeCameraDefinitionType::EYE_LOOKAT_UP_VECTORS), "Type of the camera."));

	const string ptx_path = get_path_ptx("pinhole_camera.cu");
	string camera_name = (camera_type == PinholeCameraDefinitionType::INVERSE_CAMERA_MATRIX) ? "pinhole_camera_w_matrix" : "pinhole_camera";
	// Exception / miss programs

	Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, camera_name);

	context->setRayGenerationProgram(as_integer(CameraType::STANDARD_RT), ray_gen_program);
	context->setExceptionProgram(as_integer(CameraType::STANDARD_RT), context->createProgramFromPTXFile(ptx_path, "exception"));

	set_miss_program();


	std::string ptx_path_t = get_path_ptx("tonemap_camera.cu");
	Program ray_gen_program_t = context->createProgramFromPTXFile(ptx_path_t, "tonemap_camera");

	context->setRayGenerationProgram(as_integer(CameraType::TONE_MAPPING), ray_gen_program_t);

	// Opengl Camera
	std::string ptx_path2 = get_path_ptx("opengl_camera.cu");
	Program ogl_ray_gen_program = context->createProgramFromPTXFile(ptx_path2, "opengl_camera");
	context->setRayGenerationProgram(as_integer(CameraType::TEXTURE_PASS), ogl_ray_gen_program);


	// Opengl Camera
	std::string ptx_path3 = get_path_ptx("hybrid_cameras.cu");
	Program hyb_ray_gen_program = context->createProgramFromPTXFile(ptx_path3, "hybrid_camera");
	context->setRayGenerationProgram(as_integer(CameraType::HYBRID_START), hyb_ray_gen_program);

	// Environment cameras
	bool is_env = false;
	RTsize env_tex_width, env_tex_height;
	if (default_miss == BackgroundType::ENVIRONMENT_MAP)
	{
		is_env = true;
		std::string ptx_path = get_path_ptx("env_cameras.cu");
		environment_sampler.get()->getBuffer(0, 0)->getSize(env_tex_width, env_tex_height);
		context["env_luminance"]->set(createOutputBuffer(RT_FORMAT_FLOAT, env_tex_width, env_tex_height));
		{
			Program ray_gen_program_1 = context->createProgramFromPTXFile(ptx_path, "env_luminance_camera");
			context->setRayGenerationProgram(as_integer(CameraType::ENV_1), ray_gen_program_1);
		}
		context["marginal_f"]->set(createOutputBuffer(RT_FORMAT_FLOAT, env_tex_height, 1));
		{
			Program ray_gen_program_2 = context->createProgramFromPTXFile(ptx_path, "env_marginal_camera");
			context->setRayGenerationProgram(as_integer(CameraType::ENV_2), ray_gen_program_2);
		}
		context["marginal_pdf"]->set(createOutputBuffer(RT_FORMAT_FLOAT, env_tex_height, 1));
		context["conditional_pdf"]->set(createOutputBuffer(RT_FORMAT_FLOAT, env_tex_width, env_tex_height));
		context["marginal_cdf"]->set(createOutputBuffer(RT_FORMAT_FLOAT, env_tex_height, 1));
		context["conditional_cdf"]->set(createOutputBuffer(RT_FORMAT_FLOAT, env_tex_width, env_tex_height));
		{
			Program ray_gen_program_3 = context->createProgramFromPTXFile(ptx_path, "env_pdf_camera");
			context->setRayGenerationProgram(as_integer(CameraType::ENV_3), ray_gen_program_3);
		}
	}
	else
	{
		context["marginal_pdf"]->set(createOutputBuffer(RT_FORMAT_FLOAT, 2, 1));
		context["conditional_pdf"]->set(createOutputBuffer(RT_FORMAT_FLOAT, 2, 2));
		context["marginal_cdf"]->set(createOutputBuffer(RT_FORMAT_FLOAT, 2, 1));
		context["conditional_cdf"]->set(createOutputBuffer(RT_FORMAT_FLOAT, 2, 2));
	}


	Logger::info <<"Loading camera parameters..."<<endl;
	float max_dim = m_scene_bounding_box.extent(m_scene_bounding_box.longestAxis());
	
	load_camera(init_camera_data);

	// Set ray tracing epsilon for intersection tests
	float scene_epsilon = 1.e-4f * max_dim;
	context["scene_epsilon"]->setFloat(scene_epsilon);

	// Prepare to run 
	context->validate();
	context->compile();

	// Opengl setup
	glClearColor(0.0, 0.0, 0.0, 1.0);
	float3 envmap_deltas_deg = ParameterParser::get_parameter<float3>("light", "envmap_deltas", make_float3(0), "Rotation offsetof environment map.");
	setDeltaX(&envmap_deltas_deg.x, this);
	setDeltaY(&envmap_deltas_deg.y, this);
	setDeltaZ(&envmap_deltas_deg.z, this);

	Logger::info << "Deltas << lightmap: " << to_string(envmap_deltas.x) << " " << to_string(envmap_deltas.y) << " " << to_string(envmap_deltas.z)  <<endl;
	if (is_env)
	{
		presample_environment_map();
	}

	if (gui == nullptr)
		gui = new GUI("GUI", camera->get_width(), camera->get_height());

	if (mAutoMode)
	{
		gui->toggleVisibility();
		GLUTDisplay::setContinuousMode(GLUTDisplay::CDBenchmark);
	}
	comparison_image = loadTexture(context->getContext(), "", make_float3(0));

	initUI();
	context["show_difference_image"]->setInt(show_difference_image);
	context["merl_brdf_multiplier"]->setFloat(make_float3(1));

	 Logger::info<<"Scene initialized."<<endl;
}

Matrix3x3 get_offset_lightmap_rotation_matrix(float delta_x, float delta_y, float delta_z, const optix::Matrix3x3& current_matrix)
{
	Mat3x3f matrix = rotation_Mat3x3f(ZAXIS, delta_z) * rotation_Mat3x3f(YAXIS, delta_y) * rotation_Mat3x3f(XAXIS, delta_x);
	Matrix3x3 optix_matrix = *reinterpret_cast<optix::Matrix3x3*>(&matrix);
	optix_matrix = optix_matrix * current_matrix;
	return optix_matrix;
}

void ObjScene::trace(const RayGenCameraData& s_camera_data, bool& display)
{
	context["comparison_image_weight"]->setFloat(comparison_image_weight);
	context["show_difference_image"]->setInt(show_difference_image);
	context["comparison_texture"]->setInt(comparison_image->getId());

	//Logger::debug({ "Merl correction factor: ", to_string(merl_correction.x), " ", to_string(merl_correction.y), " ", to_string(merl_correction.z) });

	camera->update_camera(s_camera_data);
	camera->set_into_gpu(context);
	context["lightmap_multiplier"]->setFloat(lightmap_multiplier);
	context["tonemap_multiplier"]->setFloat(tonemap_multiplier);
	context["tonemap_exponent"]->setFloat(tonemap_exponent);
	

	Matrix3x3 l = get_offset_lightmap_rotation_matrix(envmap_deltas.x, envmap_deltas.y, envmap_deltas.z, rotation_matrix_envmap);
	context["lightmap_rotation_matrix"]->setMatrix3x3fv(false, l.getData());

	if (m_camera_changed)
	{
		reset_renderer();
		m_camera_changed = false;
	}

	context["frame"]->setUint(m_frame++);

	double time;
	sutilCurrentTime(&time);


	if (deforming)
		scene->getAcceleration()->markDirty();

    method->pre_trace();
    execute_on_scene_elements([=](Mesh & m)
    {
        m.mShader->pre_trace_mesh(m);
    });

	// Launch the ray tracer
	Buffer buffer = context[OUTPUT_BUFFER]->getBuffer();
	RTsize buffer_width, buffer_height;
	buffer->getSize(buffer_width, buffer_height);

	unsigned int width = camera->get_width();
	unsigned int height = camera->get_height();
	context->launch(as_integer(CameraType::STANDARD_RT), width, height);

	double time1;
	sutilCurrentTime(&time1);
	// cout << "Elapsed (ray tracing): " << (time1 - time) * 1000 << endl;
	// Apply tone mapping
	if (use_tonemap)
		context->launch(as_integer(CameraType::TONE_MAPPING), width, height);


	collect_image(m_frame);
}

Buffer ObjScene::getOutputBuffer()
{
	if (use_tonemap)
		return context["tonemap_output_buffer"]->getBuffer();
	return context[OUTPUT_BUFFER]->getBuffer();
}

optix::Buffer ObjScene::createPBOOutputBuffer(const char* name, RTformat format, unsigned width, unsigned height)
{
	// Set number of devices to be used
	// Default, 0, means not to specify them here, but let OptiX use its default behavior.
	if (m_num_devices)
	{
		int max_num_devices = Context::getDeviceCount();
		int actual_num_devices = std::min(max_num_devices, std::max(1, m_num_devices));
		std::vector<int> devs(actual_num_devices);
		for (int i = 0; i < actual_num_devices; ++i) devs[i] = i;
		context->setDevices(devs.begin(), devs.end());
	}

    Buffer buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	buffer->setFormat(format);
    buffer->setSize(width, height);
    context[name]->setBuffer(buffer);

	return buffer;
}

void ObjScene::add_lights(vector<TriangleLight>& area_lights)
{
	Logger::info << "Adding light buffers to scene..." << endl;
    LightTypes::EnumType default_light_type = LightTypes::String2Enum(ParameterParser::get_parameter<string>("light", "default_light_type", LightTypes::Enum2String(LightTypes::DIRECTIONAL_LIGHT), "Type of the default light"));

	float3 light_dir = ParameterParser::get_parameter<float3>("light","default_directional_light_direction", make_float3(0.0f, -1.0f, 0.0f), "Direction of the default directional light");
	float3 light_radiance = ParameterParser::get_parameter<float3>("light", "default_directional_light_intensity", make_float3(5.0f), "Intensity of the default directional light");
	float3 light_pos = ParameterParser::get_parameter<float3>("light", "default_point_light_position", make_float3(0.08f, 0.1f, 0.11f), "Position of the default point light.");
	float3 light_intensity = ParameterParser::get_parameter<float3>("light", "default_point_light_intensity", make_float3(0.05f), "Intensity of the default point light.");
	int shadows = ParameterParser::get_parameter<int>("light", "shadows", 1, "Use shadows in rendering.");

	std::string ptx_path_light = get_path_ptx("light_programs.cu");
    Buffer dir_light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	dir_light_buffer->setFormat(RT_FORMAT_USER);
    dir_light_buffer->setElementSize(sizeof(DirectionalLight));
    dir_light_buffer->setSize(1);

    Buffer point_light_buffer = context->createBuffer(RT_BUFFER_INPUT);
    point_light_buffer->setFormat(RT_FORMAT_USER);
    point_light_buffer->setElementSize(sizeof(PointLight));
    point_light_buffer->setSize(1);

    Buffer area_light_buffer = context->createBuffer(RT_BUFFER_INPUT);
    area_light_buffer->setFormat(RT_FORMAT_USER);
    area_light_buffer->setElementSize(sizeof(TriangleLight));
    area_light_buffer->setSize(1);
    context["light_type"]->setInt(as_integer(default_light_type));
	switch (default_light_type)
	{
	case LightTypes::SKY_LIGHT:
		{
			DirectionalLight light;
			sky_model.get_directional_light(light);
            memcpy(dir_light_buffer->map(), &light, sizeof(DirectionalLight));
            dir_light_buffer->unmap();
		}
		break;
	case LightTypes::DIRECTIONAL_LIGHT:
		{
			DirectionalLight light = {normalize(light_dir), 0, light_radiance, shadows};
            memcpy(dir_light_buffer->map(), &light, sizeof(DirectionalLight));
            dir_light_buffer->unmap();
		}
		break;
	case LightTypes::POINT_LIGHT:
		{
			PointLight light = {light_pos, 0, light_intensity, shadows};
            memcpy(point_light_buffer->map(), &light, sizeof(PointLight));
            point_light_buffer->unmap();
		}
		break;
	case LightTypes::AREA_LIGHT:
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

	context["directional_lights"]->set(dir_light_buffer);
	context["point_lights"]->set(point_light_buffer);
	context["area_lights"]->set(area_light_buffer);
}

GeometryGroup ObjScene::get_geometry_group(unsigned int idx)
{
	RTobject object;
	rtGroupGetChild(scene->get(), idx, &object);
	RTgeometrygroup temp = reinterpret_cast<RTgeometrygroup>(object);
	return GeometryGroup::take(temp);
}

optix::Matrix4x4 ObjScene::get_object_transform(string filename)
{
	return optix::Matrix4x4::identity();
}


bool ObjScene::export_raw(string& raw_path)
{
	// export render data
	if (raw_path.length() <= 4 || raw_path.substr(raw_path.length() - 4).compare(".raw") != 0)
	{
		Logger::error <<  "Invalid raw file specified"<< raw_path  <<endl;
		return false;
	}
	std::string txt_file = raw_path.substr(0, raw_path.length() - 4) + ".txt";
	ofstream ofs_data(txt_file);
	if (ofs_data.bad())
	{
		Logger::error <<  "Unable to open file " << txt_file << endl;
		return false;
	}
	ofs_data << m_frame << endl << camera->get_width() << " " << camera->get_height() << endl;
	ofs_data << 1.0 << " " << 1.0f << " " << 1.0f;
	ofs_data.close();

	Buffer out = getOutputBuffer();
	int size_buffer = camera->get_width() * camera->get_height() * 4;
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

	int size_image = camera->get_width() * camera->get_height() * 3;
	float* converted = new float[size_image];
	float average = 0.0f;
	for (int i = 0; i < size_image / 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			if (!isfinite(mapped[i * 4 + j]))
			{
			}
			converted[i * 3 + j] = mapped[i * 4 + j];
			average += mapped[i * 4 + j];
		}
	}
	average /= size_image * 3;
	delete[] mapped;
	ofs_image.write(reinterpret_cast<const char*>(converted), size_image * sizeof(float));
	ofs_image.close();
	delete[] converted;
	Logger::info <<"Exported buffer to " << raw_path << " (avg: " << to_string(average) << ")" <<endl;

	return true;
}

void ObjScene::doResize(unsigned int width, unsigned int height)
{
	glutReshapeWindow(camera->get_width(), camera->get_height());
}

void ObjScene::resize(unsigned int width, unsigned int height)
{
	doResize(width, height);
}

bool ObjScene::mousePressed(int button, int state, int x, int y)
{
	if (button == 0x02 && debug_mode_enabled)
	{
		setDebugPixel(x, y);
		return true;
	}
	return gui->mousePressed(button, state, x, y);
}

bool ObjScene::mouseMoving(int x, int y)
{
	return gui->mouseMoving(x, y);
}

void ObjScene::postDrawCallBack()
{
	gui->draw();
}

void ObjScene::setDebugPixel(int i, int y)
{
	y = camera->get_height() - y;

	Logger::info <<"Setting debug pixel to " << to_string(i) << " << " << to_string(y) <<endl;
	context->setPrintLaunchIndex(i, y);
	context["debug_index"]->setUint(i, y);
}

void ObjScene::set_miss_program()
{
	string env_map_name = ParameterParser::get_parameter<string>("light", "environment_map", "pisa.hdr", "Environment map file");
	rotation_matrix_envmap = ParameterParser::get_parameter<optix::Matrix3x3>("light", "lightmap_rotation_matrix", optix::Matrix3x3::identity(), "Environment map rotation");
	lightmap_multiplier = ParameterParser::get_parameter<float3>("light", "lightmap_multiplier", make_float3(1.0f), "Environment map multiplier");
	context["lightmap_rotation_matrix"]->setMatrix3x3fv(false, rotation_matrix_envmap.getData());
	context["lightmap_multiplier"]->setFloat(lightmap_multiplier);
	context["environment_map_tex_id"]->setInt(0);

	switch (default_miss)
	{
	case BackgroundType::ENVIRONMENT_MAP:
		{
			Logger::debug<<"Loading environment map " << env_map_name << "..." <<endl;
			environment_sampler = loadTexture(context->getContext(), Folders::texture_folder + env_map_name, make_float3(1.0f));
			context["environment_map_tex_id"]->setInt(environment_sampler->getId());
			context->setMissProgram(0, context->createProgramFromPTXFile(get_path_ptx("environment_map_background.cu"), "miss"));
			context->setMissProgram(1, context->createProgramFromPTXFile(get_path_ptx("environment_map_background.cu"), "miss_shadow"));
			context["importance_sample_envmap"]->setUint(1);
			break;
		}
	case BackgroundType::SKY_MODEL:
		{
			sky_model.load_data_on_GPU(context);
			context->setMissProgram(0, context->createProgramFromPTXFile(get_path_ptx("sky_model_background.cu"), "miss"));
			context->setMissProgram(1, context->createProgramFromPTXFile(get_path_ptx("sky_model_background.cu"), "miss_shadow"));
			context["importance_sample_envmap"]->setUint(0);
			break;
	}
	case BackgroundType::CONSTANT_BACKGROUND:
	default:
		context->setMissProgram(0, context->createProgramFromPTXFile(get_path_ptx("constant_background.cu"), "miss"));
		context->setMissProgram(1, context->createProgramFromPTXFile(get_path_ptx("constant_background.cu"), "miss_shadow"));
		context->setMissProgram(2, context->createProgramFromPTXFile(get_path_ptx("constant_background.cu"), "miss"));
		context["importance_sample_envmap"]->setUint(0);
		break;
	}
}

void ObjScene::set_rendering_method(RenderingMethodType::EnumType t)
{
	switch (t)
	{
	case RenderingMethodType::RECURSIVE_RAY_TRACING:
		method = new SimpleTracing(context);
		break;
	case RenderingMethodType::AMBIENT_OCCLUSION:
		method = new AmbientOcclusion(context);
		break;
	case RenderingMethodType::PATH_TRACING:
		method = new PathTracing(context);
		break;
	default:
		Logger::error<<"The selected rendering method is not valid or no longer supported."<< endl;
		break;
	}
    method->init();
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

void ObjScene::setDebugMode(const void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);

	scene->setDebugEnabled(*(bool*)var);
}

void ObjScene::setIor(const void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	scene->global_ior_override = (*(float*)var);
	scene->updateGlassObjects();
}

void ObjScene::getIor(void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	*(float*)var = scene->global_ior_override;
}

void ObjScene::setAbsorptionColorR(const void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	scene->global_absorption_override.x = (*(float*)var);
	scene->updateGlassObjects();
}

void ObjScene::setAbsorptionColorG(const void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	scene->global_absorption_override.y = (*(float*)var);
	scene->updateGlassObjects();
}

void ObjScene::setAbsorptionColorB(const void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	scene->global_absorption_override.z = (*(float*)var);
	scene->updateGlassObjects();
}


void ObjScene::getAbsorptionColorR(void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	*(float*)var = scene->global_absorption_override.x;
}

void ObjScene::getAbsorptionColorG(void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	*(float*)var = scene->global_absorption_override.y;
}

void ObjScene::getAbsorptionColorB(void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	*(float*)var = scene->global_absorption_override.z;
}


void ObjScene::setAbsorptionInverseMultiplier(const void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	scene->global_absorption_inv_multiplier = (*(float*)var);
	scene->updateGlassObjects();
}


void ObjScene::setMedium(const void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	int c = (*(int*)var);
	MPMLMedium * m = scene->available_media[c];
	float ior = dot(m->ior_real, make_float3(0.33333f));
	float3 ab = m->absorption;
	// HDR colors...

	scene->global_ior_override = ior;
	scene->global_absorption_override = ab;
	scene->updateGlassObjects();
}


void ObjScene::setDeltaX(const void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	scene->envmap_deltas.x = *(float*)var / 180.0f * M_PI;
	scene->presample_environment_map();
	scene->reset_renderer();
}

void ObjScene::setDeltaY(const void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	scene->envmap_deltas.y = *(float*)var / 180.0f * M_PI;
	scene->presample_environment_map();
	scene->reset_renderer();
}


void ObjScene::getMedium(void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	*(int*)var = scene->current_medium;
}

void ObjScene::getAbsorptionInverseMultiplier(void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	*(float*)var = scene->global_absorption_inv_multiplier;
}

void ObjScene::getDebugMode(void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	*(bool*)var = scene->debug_mode_enabled;
}

void ObjScene::setRTDepth(const void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	int depth = *(int*)var;
	scene->context["max_depth"]->setInt(depth);
}

void ObjScene::getRTDepth(void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	*(int*)var = scene->context["max_depth"]->getInt();
}

void ObjScene::getDeltaX(void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	*(float*)var = scene->envmap_deltas.x * 180.0f / M_PI;
}

void ObjScene::getDeltaY(void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	*(float*)var = scene->envmap_deltas.y * 180.0f / M_PI;
}

void ObjScene::setDeltaZ(const void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	scene->envmap_deltas.z = *(float*)var / 180.0f * M_PI;
	scene->presample_environment_map();
	scene->reset_renderer();
}

void ObjScene::getDeltaZ(void* var, void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	*(float*)var = scene->envmap_deltas.z * 180.0f / M_PI;
}


void ObjScene::loadImage(void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	std::string filePath;
	if (Dialogs::openFileDialog(filePath))
	{
		std::cout << "Loading result image... " << filePath << std::endl;
		scene->add_result_image(filePath);
	}
}

void ObjScene::resetCameraCallback(void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	InitialCameraData i;
	scene->load_camera(i);
	GLUTDisplay::setCamera(i);
}

void ObjScene::saveRawCallback(void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	std::string filePath;
	if (Dialogs::saveFileDialog(filePath))
	{
		scene->export_raw(filePath);
	}
}

void ObjScene::startSimulationCallback(void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	scene->start_simulation();
}

void ObjScene::endSimulationCallback(void* data)
{
	ObjScene* scene = reinterpret_cast<ObjScene*>(data);
	scene->end_simulation();
}

void ObjScene::updateGlassObjects()
{
	Logger::info<<"Updating glass objects"<<endl;
	float3* abs = reinterpret_cast<float3*>(&calc_absorption);
	*abs = global_absorption_override / global_absorption_inv_multiplier;
	execute_on_scene_elements([=](Mesh& object)
		{
			object.mMaterial["ior"]->setFloat(global_ior_override);
			object.mMaterial["absorption"]->setFloat(*abs);
		});
}

// Environment importance sampling pre-pass
void ObjScene::presample_environment_map()
{
	if (environment_sampler.get() != nullptr)
	{
	
		Matrix3x3 l = get_offset_lightmap_rotation_matrix(envmap_deltas.x, envmap_deltas.y, envmap_deltas.z, rotation_matrix_envmap);
		context["lightmap_rotation_matrix"]->setMatrix3x3fv(false, l.getData());
		RTsize env_tex_width, env_tex_height;
		environment_sampler.get()->getBuffer(0, 0)->getSize(env_tex_width, env_tex_height);
		Logger::info << "Presampling envmaps... (size " << to_string(env_tex_width) << " " << to_string(env_tex_height) << ")" << endl;

		context->launch(as_integer(CameraType::ENV_1), env_tex_width, env_tex_height);
		context->launch(as_integer(CameraType::ENV_2), env_tex_width, env_tex_height);
		context->launch(as_integer(CameraType::ENV_3), env_tex_width, env_tex_height);
	}
}

void ObjScene::load_camera(InitialCameraData & camera_data)
{
	Logger::info <<"Loading camera parameters..." << endl;
	float max_dim = m_scene_bounding_box.extent(m_scene_bounding_box.longestAxis());
	float3 eye = m_scene_bounding_box.center();
	eye.z += 1.75f * max_dim;
	//*
	auto camera_type = PinholeCameraDefinitionType::String2Enum(ParameterParser::get_parameter<string>("camera", "camera_definition_type", PinholeCameraDefinitionType::Enum2String(PinholeCameraDefinitionType::EYE_LOOKAT_UP_VECTORS), "Type of the camera."));

	bool use_auto_camera = ParameterParser::get_parameter<bool>("camera", "use_auto_camera", false, "Use a automatic placed camera or use the current data.");

	Matrix3x3 camera_matrix = Matrix3x3::identity();

	fov = ParameterParser::get_parameter<float2>("camera", "camera_fov", make_float2(53.1301f, 53.1301f), "The camera FOVs (h|v)");
	if (use_auto_camera)
	{
		camera_data = InitialCameraData(eye, // eye
			m_scene_bounding_box.center(), // lookat
			make_float3(0.0f, 1.0f, 0.0f), // up
			fov.x, fov.y);
	}
	else
	{
		eye = ParameterParser::get_parameter<float3>("camera", "camera_position", make_float3(1, 0, 0), "The camera initial position");
		float3 lookat = ParameterParser::get_parameter<float3>("camera", "camera_lookat_point", make_float3(0, 0, 0), "The camera initial lookat point");
		float3 up = ParameterParser::get_parameter<float3>("camera", "camera_up", make_float3(0, 1, 0), "The camera initial up");
		camera_data = InitialCameraData(eye, lookat, up, fov.x, fov.y);
	}

	if (camera_type == PinholeCameraDefinitionType::INVERSE_CAMERA_MATRIX)
	{
		camera_matrix = ParameterParser::get_parameter<Matrix3x3>("camera", "inv_camera_matrix", Matrix3x3::identity(), "The camera inverse calibration matrix K^-1 * R^-1");
	}

	// Declare camera variables.  The values do not matter, they will be overwritten in trace.
	context["inv_calibration_matrix"]->setMatrix3x3fv(false, camera_matrix.getData());
	context["eye"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
	context["U"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
	context["V"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
	context["W"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
	reset_renderer();
}

void ObjScene::init_simulation(SimulationParameters & m_simulation_parameters)
{
	switch (m_simulation_parameters.parameter_to_simulate)
	{
	case SimulationParameters::IOR: setIor(&m_simulation_parameters.start, this); break;
	case SimulationParameters::ANGLE_X:  setDeltaX(&m_simulation_parameters.start, this);  break;
	case SimulationParameters::ANGLE_Y:  setDeltaY(&m_simulation_parameters.start, this); break;
	case SimulationParameters::ANGLE_Z:  setDeltaZ(&m_simulation_parameters.start, this); break;
	case SimulationParameters::LIGHTMAP_MULTIPLIER:  lightmap_multiplier = make_float3(m_simulation_parameters.start); break;
	case SimulationParameters::ABSORPTION_R:  global_absorption_override.x = m_simulation_parameters.start; updateGlassObjects();  break;
	case SimulationParameters::ABSORPTION_G:  global_absorption_override.y = m_simulation_parameters.start; updateGlassObjects(); break;
	case SimulationParameters::ABSORPTION_B:  global_absorption_override.z = m_simulation_parameters.start; updateGlassObjects(); break;
	case SimulationParameters::ABSORPTION_M:  global_absorption_inv_multiplier = m_simulation_parameters.start; updateGlassObjects(); break;
	case SimulationParameters::ALL_ANGLES:
	{
		m_simulation_parameters.additional_parameters = new float[3];
		getDeltaX(&m_simulation_parameters.additional_parameters[0], this);
		getDeltaY(&m_simulation_parameters.additional_parameters[1], this);
		getDeltaZ(&m_simulation_parameters.additional_parameters[2], this);
		float3 var = *reinterpret_cast<float3*>(m_simulation_parameters.additional_parameters) + make_float3(m_simulation_parameters.start);
		setDeltaX(&var.x, this);
		setDeltaY(&var.y, this);
		setDeltaZ(&var.z, this);
	}
		break;
	default: break;
	}
}

void ObjScene::update_simulation(SimulationParameters & m_simulation_parameters)
{
	float step = m_simulation_parameters.step;
	switch (m_simulation_parameters.parameter_to_simulate)
	{
	case SimulationParameters::ANGLE_X:
	{
		float var;
		getDeltaX(&var, this);
		var += step;
		setDeltaX(&var, this);
		m_simulation_parameters.status = (var > m_simulation_parameters.end) ? SimulationParameters::FINISHED : SimulationParameters::RUNNING;
	}
    break;
	case SimulationParameters::ANGLE_Y: 
	{
		float var;
		getDeltaY(&var, this);
		var += step;
		setDeltaY(&var, this);
		m_simulation_parameters.status = (var > m_simulation_parameters.end) ? SimulationParameters::FINISHED : SimulationParameters::RUNNING;
	}
	break;
	case SimulationParameters::ANGLE_Z: 	
	{
		float var;
		getDeltaZ(&var, this);
		var += step;
		setDeltaZ(&var, this);
		m_simulation_parameters.status = (var > m_simulation_parameters.end) ? SimulationParameters::FINISHED : SimulationParameters::RUNNING;
	}
	break;
	case SimulationParameters::LIGHTMAP_MULTIPLIER:	{
		lightmap_multiplier = lightmap_multiplier + make_float3(step);
		m_simulation_parameters.status = (lightmap_multiplier.x > m_simulation_parameters.end) ? SimulationParameters::FINISHED : SimulationParameters::RUNNING;
	}
	break;
	case SimulationParameters::ABSORPTION_R:
	case SimulationParameters::ABSORPTION_G:
	case SimulationParameters::ABSORPTION_B:
	case SimulationParameters::ABSORPTION_M:
	{
		float* var;
		auto param = m_simulation_parameters.parameter_to_simulate;
		var = (param == SimulationParameters::ABSORPTION_R) ? &global_absorption_override.x : 
			  (param == SimulationParameters::ABSORPTION_G) ? &global_absorption_override.y : 
			  (param == SimulationParameters::ABSORPTION_B) ? &global_absorption_override.z : 
			  (param == SimulationParameters::ABSORPTION_M) ? &global_absorption_inv_multiplier : nullptr;

		if (var != nullptr)
		{
			*var = *var + step;
			m_simulation_parameters.status = (*var > m_simulation_parameters.end) ? SimulationParameters::FINISHED : SimulationParameters::RUNNING;
			updateGlassObjects();
		}
	}
		break;
	case SimulationParameters::ALL_ANGLES: {
		float3 var;
		getDeltaX(&var.x, this);
		getDeltaY(&var.y, this);
		getDeltaZ(&var.z, this);
		var.z += step;
		float3 * initial = reinterpret_cast<float3*>(m_simulation_parameters.additional_parameters);

		float3 end = *initial + make_float3(m_simulation_parameters.end);
		float3 start = *initial + make_float3(m_simulation_parameters.start);
		m_simulation_parameters.status = SimulationParameters::RUNNING;
		if (var.z > end.z)
		{
			var.z = start.z;
			var.y += step;
			if (var.y > end.y)
			{
				var.y = start.y;
				var.x += step;
				if (var.x > end.x)
				{
					m_simulation_parameters.status = SimulationParameters::FINISHED;
					delete[] m_simulation_parameters.additional_parameters;
				}
			}
		}
		setDeltaX(&var.x, this);
		setDeltaY(&var.y, this);
		setDeltaZ(&var.z, this);
	}	
	break;
	default:
	case SimulationParameters::IOR: 	{
		float var;
		getIor(&var, this);
		var += step;
		setIor(&var, this);
		m_simulation_parameters.status = (var > m_simulation_parameters.end) ? SimulationParameters::FINISHED : SimulationParameters::RUNNING;
	}
	break;
	}
}


std::string ObjScene::get_name()
{
	float3 var;
	getDeltaX(&var.x, this);
	getDeltaY(&var.y, this);
	getDeltaZ(&var.z, this);
	return to_string(global_ior_override) + "_" + to_string(lightmap_multiplier.x) + "_" + to_string(lightmap_multiplier.y) + "_" + to_string(lightmap_multiplier.z) + "_" + available_media[current_medium]->name + "_" + to_string(global_absorption_inv_multiplier) + "_" + to_string(envmap_deltas.x) + "_" + to_string(envmap_deltas.y) + "_" + to_string(envmap_deltas.z);
}