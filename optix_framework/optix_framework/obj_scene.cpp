// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011
#ifdef NOMINMAX
#undef NOMINMAX
#endif
#include <GL/glut.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sutil.h>
#include "obj_loader.h"
#include "singular_light.h"
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
#include "GLUTDisplay.h"
#include "aisceneloader.h"
#include "shader_factory.h"
#include "Medium.h"
#include "mesh.h"
#include "host_material.h"
#include "environment_map_background.h"
#include "constant_background.h"
#include "PerlinNoise.h"
#include <algorithm>
#include "optix_utils.h"
#include "default_shader.h"
#include <sampled_bssrdf.h>

using namespace std;
using namespace optix;

#define GUI_TO_USE gui

void ObjScene::add_result_image(const string& image_file)
{
	comparison_image = loadTexture(context->getContext(), image_file, optix::make_float3(0, 1, 0));
	gui->setVisible("Weight", true);
	context["comparison_texture"]->setInt(comparison_image->getId());
}


void ObjScene::execute_on_scene_elements(function<void(Mesh&)> operation) 
{
    for (std::unique_ptr<Mesh> & m : mMeshes)
    {
        operation(*m);
    }
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

	std::string name = std::string("rendering_") + to_string(frame) + ".raw";
	export_raw(name);
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
	if (GUI_TO_USE->keyPressed(key, x, y) || key >= 48 && key <= 57) // numbers avoided
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
	case 'g':
		{
			GUI_TO_USE->toggleVisibility();
			return true;
		}
		break;
	case 'r':
	{
		Logger::info << "Realoading all shaders..." << std::endl;
		execute_on_scene_elements([=](Mesh & m)
		{
			m.reload_shader();
		});
	}
	break;
	default: return false;
	}
	return false;
}


void ObjScene::initUI()
{
	Logger::info << "Initializing UI..." << endl;

    gui->addIntVariable("Frames", reinterpret_cast<int*>(&m_frame), "Settings");
	gui->setReadOnly("Frames");
	gui->addCheckBoxCallBack("Debug Mode", setDebugMode, getDebugMode, this, "Settings");
	gui->addIntVariableCallBack("Depth", setRTDepth, getRTDepth, this, "Settings", 0, 100000, 1);

	//const char * t comp_group = "Comparison";
	//gui->addButton("Load result image", loadImage, this, comp_group);
	//gui->addFloatVariable("Weight", &comparison_image_weight, comp_group, 0.0f, 1.0f, 0.01f);
	//gui->addCheckBox("Difference image", &show_difference_image, comp_group);
	//gui->setVisible("Weight", false);

	const char* tmg = "Tone mapping";
	gui->addCheckBox("Tonemap", &use_tonemap, tmg);
	gui->addFloatVariable("Multiplier", &tonemap_multiplier, tmg, 0.0f, 1.0f, 0.05f);
	gui->addFloatVariable("Exponent", &tonemap_exponent, tmg, 0.5f, 3.5f, 0.05f);

	//const char* glass_group = "Glass";
	//gui->addFloatVariableCallBack("Index of refraction", setIor, getIor, this, glass_group);
	//gui->addFloatVariableCallBack("Absorption - R", setAbsorptionColorR, getAbsorptionColorR, this, glass_group);
	//gui->addFloatVariableCallBack("Absorption - G", setAbsorptionColorG, getAbsorptionColorG, this, glass_group);
	//gui->addFloatVariableCallBack("Absorption - B", setAbsorptionColorB, getAbsorptionColorB, this, glass_group);
	//gui->addFloatVariableCallBack("Absorption inv. multiplier", setAbsorptionInverseMultiplier, getAbsorptionInverseMultiplier, this, glass_group);

	vector<GuiDropdownElement> elems;
	int count = 0;
	for (const auto& kv : available_media)
	{
		GuiDropdownElement e = { count++, kv->name.c_str() };
		elems.push_back(e);
	}
//	gui->addDropdownMenuCallback("Medium (glass)", elems, setMedium, getMedium, this, glass_group);

	const char* env_map_correction_group = "Environment map corrections";

	gui->addButton("Reset Camera", resetCameraCallback, this, "Settings");
	gui->addButton("Save RAW File", saveRawCallback, this, "Settings");

    miss_program->set_into_gui(gui);

	gui->addCheckBox("Use heterogenous materials", (bool*)&use_heterogenous_materials, "Settings");

    execute_on_scene_elements([=](Mesh & m)
    {
        m.set_into_gui(gui);
    });

    gui->addFloatVariableCallBack("Noise freq.",
        [](const void* var, void* data)
    {
        ObjScene* s = reinterpret_cast<ObjScene*>(data);
        s->noise_frequency = *((float*)var);
        s->create_3d_noise(s->noise_frequency);
    },
        [](void* var, void* data)
    {
        ObjScene* s = reinterpret_cast<ObjScene*>(data);
        *((float*)var) = s->noise_frequency;
    }, this, "Noise");

    camera->set_into_gui(gui);
}

void ObjScene::create_3d_noise(float frequency)
{
    static TextureSampler sampler = context->createTextureSampler();
    static optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 256u, 256u, 256u);
    static PerlinNoise p(1337);

    sampler->setWrapMode(0, RT_WRAP_REPEAT);
    sampler->setWrapMode(1, RT_WRAP_REPEAT);
    sampler->setWrapMode(2, RT_WRAP_REPEAT);
    sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
    sampler->setMaxAnisotropy(1.0f);
    sampler->setMipLevelCount(1u);
    sampler->setArraySize(1u);
    // Create buffer with single texel set to default_color
    float* buffer_data = static_cast<float*>(buffer->map());

    //for (int i = 0; i < 256; i++)
    //    for (int j = 0; j < 256; j++)
    //        for (int k = 0; k < 256; k++)
    //        {
    //            int idx = 256 * 256 * i + 256 * j + k;
    //            buffer_data[idx] = (float)p.noise(i / (256.0f) * frequency, j / (256.0f) * frequency, k / (256.0f) * frequency);
    //        }
    buffer->unmap();

    sampler->setBuffer(0u, 0u, buffer);
    context["noise_tex"]->setInt(sampler->getId());
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

    auto camera_type = PinholeCameraDefinitionType::String2Enum(ParameterParser::get_parameter<string>("camera", "camera_definition_type", PinholeCameraDefinitionType::Enum2String(PinholeCameraDefinitionType::EYE_LOOKAT_UP_VECTORS), "Type of the camera."));


    int camera_width = ParameterParser::get_parameter<int>("camera", "window_width", 512, "The width of the window");
    int camera_height = ParameterParser::get_parameter<int>("camera", "window_height", 512, "The height of the window");
    int downsampling = ParameterParser::get_parameter<int>("camera", "camera_downsampling", 1, "");
    camera = std::make_unique<Camera>(context, camera_type, camera_width, camera_height, downsampling, custom_rr);
    
    ShaderFactory::init(context);
    ShaderInfo info = {"volume_shader_heterogenous.cu", "Volume path tracer (het.)", 13};
    ShaderFactory::add_shader<DefaultShader>(info);

	ShaderInfo info2 = { "subsurface_scattering_sampled.cu", "Sampled BSSRDF", 14 };
	ShaderFactory::add_shader<SampledBSSRDF>(info2);

    for (auto& kv : MaterialLibrary::media)
	{
		available_media.push_back(&kv.second);
	}


	default_miss = BackgroundType::String2Enum(ParameterParser::get_parameter<string>("config", "default_miss_type", BackgroundType::Enum2String(BackgroundType::CONSTANT_BACKGROUND), "Default miss program."));

    tonemap_exponent = ParameterParser::get_parameter<float>("tonemap", "tonemap_exponent", 1.8f, "Tonemap exponent");
    tonemap_multiplier = ParameterParser::get_parameter<float>("tonemap", "tonemap_multiplier", 1.f, "Tonemap multiplier");

	// Setup context
	context->setRayTypeCount(RAY_TYPE_COUNT);
	context->setStackSize(ParameterParser::get_parameter<int>("config", "stack_size", 2000, "Allocated stack size for context"));
	context["use_heterogenous_materials"]->setInt(use_heterogenous_materials);

	context["max_depth"]->setInt(ParameterParser::get_parameter<int>("config", "max_depth", 5, "Maximum recursion depth of the raytracer"));
	context[OUTPUT_BUFFER]->set(createPBOOutputBuffer(OUTPUT_BUFFER, RT_FORMAT_FLOAT4, camera->get_width(), camera->get_height()));

	// Constant colors
	context["bad_color"]->setFloat(0.0f, 1.0f, 0.0f);
    context["bg_color"]->setFloat(0.3f, 0.3f, 0.3f);
	
	bool use_abs = ParameterParser::get_parameter<bool>("config", "use_absorption", true, "Use absorption in rendering.");
	Logger::debug << "Absorption is " << (use_abs ? "ON" : "OFF") << endl;

	// Tone mapping pass
	context["tonemap_output_buffer"]->set(createPBOOutputBuffer("tonemap_output_buffer", RT_FORMAT_FLOAT4, camera->get_width(), camera->get_height()));

	// Create group for scene objects and float acceleration structure
	scene = context->createGroup();
	scene->setChildCount(static_cast<unsigned int>(filenames.size()));
	Acceleration acceleration = context->createAcceleration("Trbvh");
	scene->setAcceleration(acceleration);
	acceleration->markDirty();

	// We need the scene bounding box for placing the camera
	Aabb bbox;

	
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
        vector<std::unique_ptr<Mesh>> v = loader->load(get_object_transform(filenames[i]));
		for (auto& c : v)
		{
			mMeshes.push_back(std::move(c));
		}

	    m_scene_bounding_box.include(loader->getSceneBBox());
		loader->getAreaLights(lights);
 
		delete loader;
		// Set material shaders

		// Add geometry group to the group of scene objects
		scene->setChild(i, geometry_group);
	}

    execute_on_scene_elements([=](Mesh & m)
    {
        m.set_method(t);
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

    context["top_object"]->set(scene);
    context["top_shadower"]->set(scene);

	// Add light sources depending on chosen shader
    set_miss_program();

	add_lights(lights);


	Logger::info << "Loading programs..." << endl;
	// Set top level geometry in acceleration structure. 
	// The default used by the ObjLoader is SBVH.

	// Set up camera
	
    
	std::string ptx_path_t = get_path_ptx("tonemap_camera.cu");
	Program ray_gen_program_t = context->createProgramFromPTXFile(ptx_path_t, "tonemap_camera");

	context->setRayGenerationProgram(as_integer(CameraType::TONE_MAPPING), ray_gen_program_t);

	// Environment cameras
	

	Logger::info <<"Loading camera parameters..."<<endl;
	float max_dim = m_scene_bounding_box.extent(m_scene_bounding_box.longestAxis());
	
    create_3d_noise(noise_frequency);

    load_camera_extrinsics(init_camera_data);

	// Set ray tracing epsilon for intersection tests
	float scene_epsilon = 1.e-4f * max_dim;
	context["scene_epsilon"]->setFloat(scene_epsilon);
	// Prepare to run 
	context->validate();
	context->compile();

	if (gui == nullptr)
		gui = new GUI("GUI", camera->get_width(), camera->get_height());
	if (new_gui == nullptr)
	{
		new_gui = new ImmediateGUI("GUI", camera->get_width(), camera->get_height());
	}

	if (mAutoMode)
	{
		gui->toggleVisibility();
		GLUTDisplay::setContinuousMode(GLUTDisplay::CDBenchmark);
	}
	comparison_image = loadTexture(context->getContext(), "", make_float3(0));

    MaterialDataCommon params;

    params.absorption = make_float3(0);
    params.emissive = make_float3(0);
    params.illum = 12;
    params.ior = 1.3f;
    params.phong_exp = 0;
    params.reflectivity = make_float3(0);
    params.ambient_map = loadTexture(m_context, "", make_float3(0))->getId();
    params.diffuse_map = loadTexture(m_context, "", make_float3(1, 0, 0))->getId();
    params.specular_map = loadTexture(m_context, "", make_float3(0))->getId();

    material_ketchup = std::make_shared<MaterialHost>("ketchup", params);
    execute_on_scene_elements([=](Mesh & m)
    {
        m.add_material(material_ketchup);
    });

	initUI();
	context["show_difference_image"]->setInt(show_difference_image);
	context["merl_brdf_multiplier"]->setFloat(make_float3(1));

 

	 Logger::info<<"Scene initialized."<<endl;
}



void ObjScene::trace(const RayGenCameraData& s_camera_data, bool& display)
{
	display = true;
	context["comparison_image_weight"]->setFloat(comparison_image_weight);
	context["show_difference_image"]->setInt(show_difference_image);
	context["comparison_texture"]->setInt(comparison_image->getId());
	context["use_heterogenous_materials"]->setInt(use_heterogenous_materials);

	//Logger::debug({ "Merl correction factor: ", to_string(merl_correction.x), " ", to_string(merl_correction.y), " ", to_string(merl_correction.z) });

	camera->update_camera(s_camera_data);
	camera->set_into_gpu(context);
    miss_program->set_into_gpu(context);

    context["tonemap_multiplier"]->setFloat(tonemap_multiplier);
	context["tonemap_exponent"]->setFloat(tonemap_exponent);
	
	execute_on_scene_elements([=](Mesh & m)
	{
		m.load();
	});

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
        m.pre_trace();
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

    Buffer dir_light_buffer = create_buffer<SingularLightData>(context);
    Buffer area_light_buffer = create_buffer<TriangleLight>(context);

    context["light_type"]->setInt(as_integer(default_light_type));
	switch (default_light_type)
	{
	case LightTypes::SKY_LIGHT:
		{
			SingularLightData light;
			static_cast<SkyModel*>(miss_program.get())->get_directional_light(light);
            memcpy(dir_light_buffer->map(), &light, sizeof(SingularLightData));
            dir_light_buffer->unmap();
		}
		break;
	case LightTypes::DIRECTIONAL_LIGHT:
		{
            SingularLightData light = { normalize(light_dir), LIGHT_TYPE_DIR, light_radiance, shadows };
            memcpy(dir_light_buffer->map(), &light, sizeof(SingularLightData));
            dir_light_buffer->unmap();
		}
		break;
	case LightTypes::POINT_LIGHT:
		{
            SingularLightData light = { light_pos, LIGHT_TYPE_POINT, light_intensity, shadows };
            memcpy(dir_light_buffer->map(), &light, sizeof(SingularLightData));
            dir_light_buffer->unmap();
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

	context["singular_lights"]->set(dir_light_buffer);
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
    if (raw_path.length() == 0)
    {
        Logger::error << "Invalid raw file specified" << raw_path << endl;
        return false;
    }

	if (raw_path.length() <= 4 || raw_path.substr(raw_path.length() - 4).compare(".raw") != 0)
	{
        raw_path += ".raw";
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
	return GUI_TO_USE->mousePressed(button, state, x, y);
}

bool ObjScene::mouseMoving(int x, int y)
{
	return GUI_TO_USE->mouseMoving(x, y);
}

void ObjScene::postDrawCallBack()
{
	GUI_TO_USE->draw();
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
   
	switch (default_miss)
	{
	case BackgroundType::ENVIRONMENT_MAP:
	{
        string env_map_name = ParameterParser::get_parameter<string>("light", "environment_map", "pisa.hdr", "Environment map file");
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
        float3 color = ParameterParser::get_parameter<float3>("light", "background_constant_color", make_float3(0.5), "Environment map file");
        miss_program = std::make_unique<ConstantBackground>(color);
        break;
    }
    miss_program->init(context);
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
		//context->setExceptionEnabled(RT_EXCEPTION_ALL, true);
	}
	else
	{
		context->setPrintEnabled(false);
	//	context->setExceptionEnabled(RT_EXCEPTION_ALL, false);
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
	scene->load_camera_extrinsics(i);
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


void ObjScene::load_camera_extrinsics(InitialCameraData & camera_data)
{
    auto camera_type = PinholeCameraDefinitionType::String2Enum(ParameterParser::get_parameter<string>("camera", "camera_definition_type", PinholeCameraDefinitionType::Enum2String(PinholeCameraDefinitionType::EYE_LOOKAT_UP_VECTORS), "Type of the camera."));  

	float max_dim = m_scene_bounding_box.extent(m_scene_bounding_box.longestAxis());
	float3 eye = m_scene_bounding_box.center();
	eye.z += 3 * max_dim;

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

	reset_renderer();
}
