#pragma once
#include "SampleScene.h"
#include <memory>
#include <bssrdf_creator.h>
#include <bssrdf_loader.h>
#include <reference_bssrdf_gpu.h>
#include <map>
#include "string_utils.h"
#include "logger.h"
#include "render_task.h"

#define INVALID_INDEX ((size_t)(-1))
class ImmediateGUI;

struct ParameterState
{
	ParameterState() {}
	ParameterState(const std::vector<size_t>& data) : mData(data)  {}
	
	std::string tostring()
	{
		std::string res = "(";
		for (int i = 0; i < mData.size(); i++)
		{
			res += std::to_string(mData[i]) + ((i == mData.size() - 1) ? "" : " ");
		}
		return res + ")";
	}

	const size_t& operator[](const size_t & idx) const 
	{		
		if (idx >= 8)
			Logger::error << "Out of bounds!" << std::endl;
		return mData[idx];
	}

	size_t& operator[](const size_t & idx) 
	{
		if (idx >= mData.size())
			Logger::error << "Out of bounds!" << std::endl;
		return mData[idx];
	}

	bool operator==(const ParameterState &b) const
	{
		bool equal = true;
		for (int i = 0; i < mData.size(); i++)
		{
			equal &= b.mData[i] == mData[i];
		}
		return equal;
	}

	std::vector<size_t> mData;
};

class FullBSSRDFParameters
{
public:

	std::map<size_t, std::vector<float>> parameters = {
		{ theta_i_index,		{ 0, 15, 30, 45, 60, 70, 80, 88 } },
		{ r_index,				{ 0.01f, 0.05f, 0.1f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 2.0f, 4.0f, 8.0f, 10.0f } },
		{ theta_s_index,		{ 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180 } },
		{ albedo_index,			{ 0.01f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.99f } },
		{ g_index,				{ -0.9f, -0.7f, -0.5f, -0.3f, 0.0f, 0.3f, 0.5f, 0.7f, 0.9f, 0.95f, 0.99f } },
		{ eta_index,			{1.0f, 1.1f, 1.2f, 1.3f, 1.4f } }
	};

	//std::map<size_t, std::vector<float>> parameters = {
	//	{ theta_i_index,	{ 0, 15, 30, 45, 60, 70, 80, 88 } },
	//	{ r_index,			{ 0.01f, 0.05f, 0.1f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 2.0f, 4.0f, 8.0f, 10.0f } },
	//	{ theta_s_index,	{ 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180 } },
	//	{ albedo_index,		{ 0.9f } },
	//	{ g_index,			{ 0.3f } },
	//	{ eta_index,		{ 1.0f } }
	//};

#define stringify_pair(x) x, #x
	std::map<size_t, std::string> parameter_names = {
		{ stringify_pair(theta_i_index) },
		{ stringify_pair(r_index) },
		{ stringify_pair(theta_s_index) },
		{ stringify_pair(albedo_index) },
		{ stringify_pair(g_index) },
		{ stringify_pair(eta_index) }
	};
#undef stringify_pair

	size_t size()
	{
		std::vector<size_t> dims = get_dimensions();
		size_t res = 1;
		for (const size_t& i : dims)
			res *= i;
		return res;
	}

	void get_parameters(const ParameterState & state, float & theta_i, float & r, float & theta_s, float & albedo, float & g, float & eta)
	{
		theta_i = parameters[theta_i_index][state[theta_i_index]];
		r = parameters[r_index][state[r_index]];
		theta_s = parameters[theta_s_index][state[theta_s_index]];
		albedo = parameters[albedo_index][state[albedo_index]];
		g = parameters[g_index][state[g_index]];
		eta = parameters[eta_index][state[eta_index]];
	}

	ParameterState next(const ParameterState & state)
	{
		ParameterState val = state;
		std::vector<size_t> dims = get_dimensions();
		size_t i;
		for (i = dims.size() - 1; i >= 0; i--)
		{
			// increment returns true if overflow, so we keep adding.
			if (!increment(state[i], dims[i], val[i]))
			{
				break;
			}
		}

		// When the last index overflows.
		if (i == -1)
		{
			return invalid_index;
		}
		return val;
	}

	bool is_valid(const ParameterState & state)
	{
		return !(state == invalid_index);
	}

	size_t get_size()
	{
		size_t r = 1;
		for (auto c : get_dimensions())
		{
			r *= c;
		}
		return r;
	}


	std::vector<size_t> get_dimensions();

private:
	bool increment(size_t src, size_t size, size_t & dst)
	{
		dst = (src + 1) % size;
		return ((src + 1) / size) >= 1;
	}

	const ParameterState invalid_index = ParameterState({ INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX });
};


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
	std::unique_ptr<ImmediateGUI> gui;

	FullBSSRDFParameters mParameters;
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

