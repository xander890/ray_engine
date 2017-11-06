#pragma once
#include "SampleScene.h"
#include <memory>
#include <bssrdf_creator.h>
#include <bssrdf_loader.h>
#include <reference_bssrdf_gpu.h>
#include "string_utils.h"
#include "logger.h"
#include "render_task.h"
#include "bssrdf_parameter_manager.h"
#include "full_bssrdf_host_device_common.h"
class ImmediateGUI;


class FullBSSRDFGenerator : public SampleScene
{
public:

	enum RenderMode { RENDER_BSSRDF = 0, SHOW_EXISTING_BSSRDF = 1};

	FullBSSRDFGenerator(const char * config, bool offline_render);
	~FullBSSRDFGenerator();
	void initialize_scene(GLFWwindow * window, InitialCameraData& camera_data) override;

	// Update camera shader with new viewing params and then trace
	void trace(const RayGenCameraData& camera_data) override;

	// Return the output buffer to be displayed
	virtual optix::Buffer get_output_buffer();

	void post_draw_callback() override;

	void start_rendering();
	void update_rendering(float deltaTime);
	void end_rendering();

	bool key_pressed(int key, int x, int y) override;
	bool mouse_pressed(int x, int y, int button, int action, int mods) override;
	bool mouse_moving(int x, int y) override;

	void clean_up() override;
	void scene_initialized() override;

	void set_render_task(std::unique_ptr<RenderTask>& task);
private:

	void set_external_bssrdf(const std::string & file);

	optix::Buffer result_buffer;
	std::shared_ptr<BSSRDFRenderer>	 mCurrentBssrdfRenderer;
	std::shared_ptr<ReferenceBSSRDFGPU>	 mBssrdfReferenceSimulator;
	std::shared_ptr<BSSRDFRendererModel>		 mBssrdfModelSimulator;
	std::string config_file;

	int entry_point_output = -1;
	optix::Buffer mBSSRDFBufferTexture = nullptr;
	optix::TextureSampler mBSSRDFHemisphereTex = nullptr;
	float mScaleMultiplier = 1.f;
	int mShowFalseColors = 1;
	int mFresnelMode = BSSRDF_RENDER_MODE_FULL_BSSRDF;
	std::unique_ptr<ImmediateGUI> gui;

	BSSRDFParameterManager mParametersSimulation;
	BSSRDFParameterManager mParametersOriginal;
	ParameterState mState;

	int mSimulationSamplesPerFrame = (int)1e7;
	int mSimulationMaxFrames = 100;
	int mSimulationMaxIterations = (int)1e9;

	float * mCurrentHemisphereData = nullptr;
	bool mPaused = false;
	bool mFastMode = false;
	RenderMode mCurrentRenderMode = RENDER_BSSRDF;
	std::unique_ptr<BSSRDFExporter> mExporter = nullptr;
	std::unique_ptr<BSSRDFLoader> mLoader = nullptr;

	optix::Buffer mExternalBSSRDFBuffer;
	std::string mExternalFilePath = "test.bssrdf";

	std::unique_ptr<RenderTask> current_render_task;
	bool mSimulate = 1;
	void set_render_mode(RenderMode m, bool isSimulated);
	bool start_offline_rendering = false;
};

