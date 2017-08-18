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

//#define IOR_EST


#ifdef IOR_EST
#include "indexofrefractionmatcher.h"
#endif
#include "gui.h"
#include "structs.h"
#include <functional>
#include "camera.h"
#include "camera_host.h"

class ObjScene : public SampleScene
{
public:


	ObjScene(const std::vector<std::string>& obj_filenames, const std::string& shader_name, const std::string& config_file, optix::int4 rendering_r = make_int4(-1))
        : context(m_context),
          current_scene_type(Scene::OPTIX_ONLY), default_miss(), filenames(obj_filenames), method(nullptr),        m_frame(0u),
          deforming(false),
          use_tonemap(true), 
          config_file(config_file)
    {
        calc_absorption[0] = calc_absorption[1] = calc_absorption[2] = 0.0f;
		custom_rr = rendering_r;
        mMeshes.clear();
    }

	ObjScene()
		: context(m_context),
		current_scene_type(Scene::OPTIX_ONLY), filenames(1, "test.obj"), 
		  m_frame(0u),
		  deforming(true),
		  use_tonemap(true),
		  config_file("config.xml")
	{
		calc_absorption[0] = calc_absorption[1] = calc_absorption[2] = 0.0f;
        mMeshes.clear();
		custom_rr = make_int4(-1);
	}

	virtual ~ObjScene()
	{
		ParameterParser::free();
		cleanUp();
	}

	virtual void cleanUp()
	{
		context->destroy();
		m_context = 0;
	}

	void initUI();

    void create_3d_noise(float frequency);
    float noise_scale = 1;
    float noise_frequency = 5;

	virtual void initScene(InitialCameraData& camera_data) override;
	virtual void trace(const RayGenCameraData& camera_data, bool& display) override;

	virtual void trace(const RayGenCameraData& camera_data) override
	{
		bool display = true;
		trace(camera_data, display);
	}

	void start_simulation();
	void end_simulation();

	void collect_image(unsigned int frame);
	virtual bool keyPressed(unsigned char key, int x, int y) override;
	virtual optix::Buffer getOutputBuffer() override;
	virtual void doResize(unsigned int width, unsigned int height) override;
	virtual void resize(unsigned int width, unsigned int height) override;
	void postDrawCallBack() override;
	void setDebugPixel(int i, int y);
	bool mousePressed(int button, int state, int x, int y) override;
	bool mouseMoving(int x, int y) override;
	void reset_renderer();

#ifdef IOR_EST
	IndexMatcher * index_matcher;
#endif

	virtual void getDebugText(std::string& text, float& x, float& y) override
	{
		text = "";// Scene::Enum2String(current_scene_type);
		x = 10.0f;
		y = 36.0f;
	}

	void setAutoMode();
	void setOutputFile(const string& cs);
	void setFrames(int frames);

private:
	Context context;
	bool debug_mode_enabled = true;

	Scene::EnumType current_scene_type;

	BackgroundType::EnumType default_miss;
	bool collect_images = false;
	bool show_difference_image = false;
	Aabb m_scene_bounding_box;
	bool mAutoMode = false;
	string mOutputFile = "rendering.raw";
	int mFrames = -1;
	optix::Buffer createPBOOutputBuffer(const char* name, RTformat format, unsigned width, unsigned height);

	void add_lights(vector<TriangleLight>& area_lights);
	void set_miss_program();
	optix::TextureSampler environment_sampler;
    std::unique_ptr<MissProgram> miss_program;

	optix::float2 fov;

	// Geometry and transformation getters
	optix::GeometryGroup get_geometry_group(unsigned int idx);
	optix::Matrix4x4 get_object_transform(std::string filename);

	bool export_raw(std::string& name);
	void set_rendering_method(RenderingMethodType::EnumType t);
	std::vector<std::string> filenames;

	Group scene;
	//std::vector<optix::uint2> lights;
	RenderingMethod* method;

	unsigned int m_frame;
	bool deforming;
	bool use_tonemap;
	std::unique_ptr<Camera> camera = nullptr;
   

	GUI* gui = nullptr;
	void add_result_image(const string& image_file);
    std::vector<Mesh> mMeshes;
    std::shared_ptr<MaterialHost> material_ketchup;

    void execute_on_scene_elements(function<void(Mesh&)> operation);

	void setDebugEnabled(bool var);

	static void GUI_CALL setDebugMode(const void* var, void* data);
	static void GUI_CALL getDebugMode(void* var, void* data);

	static void GUI_CALL setRTDepth(const void* var, void* data);
	static void GUI_CALL getRTDepth(void* var, void* data);

	static void GUI_CALL setIor(const void* var, void* data);
	static void GUI_CALL getIor(void* var, void* data);
	static void GUI_CALL setAbsorptionColorR(const void* var, void* data);
	static void GUI_CALL setAbsorptionColorG(const void* var, void* data);
	static void GUI_CALL setAbsorptionColorB(const void* var, void* data);
	static void GUI_CALL getAbsorptionColorR(void* var, void* data);
	static void GUI_CALL getAbsorptionColorG(void* var, void* data);
	static void GUI_CALL getAbsorptionColorB(void* var, void* data);
	static void GUI_CALL setAbsorptionInverseMultiplier(const void* var, void* data);
	static void GUI_CALL getAbsorptionInverseMultiplier(void* var, void* data);
	static void GUI_CALL setMedium(const void* var, void* data);
	static void GUI_CALL getMedium(void* var, void* data);

	static void GUI_CALL loadImage(void* data);
	static void GUI_CALL resetCameraCallback(void* data);
	static void GUI_CALL saveRawCallback(void* data);
	static void GUI_CALL startSimulationCallback(void* data);
	static void GUI_CALL endSimulationCallback(void* data);

	float tonemap_multiplier = 1.0f;
	float tonemap_exponent = 1.0f;

	optix::float3 global_absorption_override = optix::make_float3(0.0f);
	float global_absorption_inv_multiplier = 1.0f;
	float global_ior_override = 1.5f;
	float calc_absorption[3];
	void updateGlassObjects();
	std::vector<MPMLMedium*> available_media;
	int current_medium = 0;


	int mtx_method = 1;
	
	optix::TextureSampler comparison_image;
	float comparison_image_weight = 0.0;

	void load_camera_extrinsics(InitialCameraData & data);

	struct SimulationParameters
	{
		float step = 0.01f;
		float start = 1.5f;
		float * additional_parameters;
		float end = 1.7f;
		int samples = 1000;
		enum SimulationElement { IOR, ABSORPTION_R, ABSORPTION_G, ABSORPTION_B, ABSORPTION_M } parameter_to_simulate = IOR;
		enum SimulationStatus {RUNNING, FINISHED} status = RUNNING;

	} m_simulation_parameters;


	void update_simulation(SimulationParameters & m_simulation_parameters);
	void init_simulation(::ObjScene::SimulationParameters& m_simulation_parameters);
	std::string get_name();

	const std::string config_file = "config.xml";
	int4 custom_rr;
};

#endif // OBJSCENE_H
