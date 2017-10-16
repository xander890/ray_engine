#pragma once
#include "SampleScene.h"
#include <memory>
#include <bssrdf_creator.h>
#include <bssrdf_loader.h>

class ImmediateGUI;

struct ParameterState
{
	ParameterState() {}
	ParameterState(size_t theta_i_idx, size_t r_idx, size_t theta_s_idx, size_t albedo_idx, size_t g_idx, size_t eta_idx) :
		theta_i_idx(theta_i_idx), r_idx(r_idx), theta_s_idx(theta_s_idx), albedo_idx(albedo_idx), g_idx(g_idx), eta_idx(eta_idx) {}
	
	std::string tostring()
	{
		return std::string("(") + std::to_string(theta_i_idx) + "," + std::to_string(r_idx) + "," + std::to_string(theta_s_idx) + ","
			+ std::to_string(albedo_idx) + "," + std::to_string(g_idx) + "," + std::to_string(eta_idx) + ")";
	}

	size_t theta_i_idx = 0, r_idx = 0, theta_s_idx = 0, albedo_idx = 0, g_idx = 0, eta_idx = 0;
};

class FullBSSRDFParameters
{
public:
	//std::vector<float> theta_i =	{0, 15, 30, 45, 60, 70, 80, 88};
	//std::vector<float> r =			{ 0.01f, 0.05f, 0.1f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 2.0f, 4.0f, 8.0f, 10.0f };
	//std::vector<float> theta_s =	{ 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180 };
	//std::vector<float> albedo =		{ 0.01f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.99f };
	//std::vector<float> g =			{ -0.9f, -0.7f, -0.5f, -0.3f, 0.0f, 0.3f, 0.5f, 0.7f, 0.9f, 0.95f, 0.99f };
	//std::vector<float> eta =		{ 1.0f, 1.1f, 1.2f, 1.3f, 1.4f };

	//std::vector<float> theta_i_v = { 0, 15, 30, 45, 60, 70, 80, 88 };
	//std::vector<float> r_v = { 0.01f, 0.05f, 0.1f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 2.0f, 4.0f, 8.0f, 10.0f };
	//std::vector<float> theta_s_v = { 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180 };
	//std::vector<float> albedo_v = { 0.9f };
	//std::vector<float> g_v = { 0.3f };
	//std::vector<float> eta_v = { 1.0f };

	std::vector<float> theta_i_v = { 0 };
	std::vector<float> r_v = { 0.1f };
	std::vector<float> theta_s_v = { 0 };
	std::vector<float> albedo_v = { 0.9f };
	std::vector<float> g_v = { 0.3f };
	std::vector<float> eta_v = { 1.0f };

	size_t flatten(size_t theta_i_idx, size_t r_idx, size_t theta_s_idx)
	{
		size_t idx;
		idx = theta_s_idx * r_v.size() + r_idx;
		idx = idx * theta_i_v.size() + theta_i_idx;
		return idx;
	}

	size_t flatten(size_t theta_i_idx, size_t r_idx, size_t theta_s_idx, size_t albedo_idx, size_t g_idx, size_t eta_idx)
	{
		size_t idx;
		idx = g_idx * eta_v.size() + eta_idx;
		idx = idx *   albedo_v.size() + albedo_idx;
		idx = idx *   theta_s_v.size() + theta_s_idx;
		idx = idx *   r_v.size() + r_idx;
		idx = idx *   theta_i_v.size() + theta_i_idx;
		return idx;
	}

	size_t flatten(const ParameterState & state)
	{
		return flatten(state.theta_i_idx, state.r_idx, state.theta_s_idx, state.albedo_idx, state.g_idx, state.eta_idx);
	}

	size_t size()
	{
		return eta_v.size()* g_v.size()*albedo_v.size()*theta_s_v.size()*r_v.size()*theta_i_v.size();
	}

	void get_parameters(const ParameterState & state, float & theta_i, float & r, float & theta_s, float & albedo, float & g, float & eta)
	{
		theta_i = theta_i_v[state.theta_i_idx];
		r = r_v[state.r_idx];
		theta_s = theta_s_v[state.theta_s_idx];
		albedo = albedo_v[state.albedo_idx];
		g = g_v[state.g_idx];
		eta = eta_v[state.eta_idx];
	}


	ParameterState next(const ParameterState & state)
	{
		ParameterState val = state;
		if (increment(state.theta_i_idx, theta_i_v.size(), val.theta_i_idx))
			if (increment(state.r_idx, r_v.size(), val.r_idx))
				if (increment(state.theta_s_idx, theta_s_v.size(), val.theta_s_idx))
					if (increment(state.albedo_idx, albedo_v.size(), val.albedo_idx))
						if (increment(state.g_idx, g_v.size(), val.g_idx))
							if (increment(state.eta_idx, eta_v.size(), val.eta_idx))
								return invalid_index;
		return val;
	}

	bool is_valid(const ParameterState & state)
	{
		return !(state.theta_i_idx == ((size_t)-1) && state.r_idx == ((size_t)-1) && state.theta_s_idx == ((size_t)-1) && state.albedo_idx == ((size_t)-1) && state.g_idx == ((size_t)-1) && state.eta_idx == ((size_t)-1));
	}

	std::vector<size_t> get_dimensions();

private:
	bool increment(size_t src, size_t size, size_t & dst)
	{
		dst = (src + 1) % size;
		return ((src + 1) / size) >= 1;
	}
	const ParameterState invalid_index = ParameterState(-1,-1,-1,-1,-1,-1);
};


class FullBSSRDFGenerator : public SampleScene
{
public:

	enum RenderMode { RENDER_BSSRDF = 0, SHOW_EXISTING_BSSRDF = 1 };

	FullBSSRDFGenerator(const char * config);
	~FullBSSRDFGenerator();
	void initialize_scene(GLFWwindow * window, InitialCameraData& camera_data) override;

	// Update camera shader with new viewing params and then trace
	void trace(const RayGenCameraData& camera_data) override;

	// Return the output buffer to be displayed
	virtual optix::Buffer get_output_buffer();

	void post_draw_callback() override;

	void start_rendering();
	void update_rendering();
	void end_rendering();

	bool key_pressed(int key, int x, int y) override;
	bool mouse_pressed(int x, int y, int button, int action, int mods) override;
	bool mouse_moving(int x, int y) override;

	void clean_up() override;

private:

	void set_external_bssrdf(const std::string & file);

	optix::Buffer result_buffer;
	std::unique_ptr<ReferenceBSSRDF> creator;
	std::string config_file;

	int entry_point_output = -1;
	optix::Buffer mBSSRDFBufferTexture;
	optix::TextureSampler mBSSRDFHemisphereTex;
	float mScaleMultiplier = 1.f;
	int mShowFalseColors = 1;
	std::unique_ptr<ImmediateGUI> gui;
	bool is_rendering = false;

	FullBSSRDFParameters mParameters;
	ParameterState mState;
	int mSimulationCurrentFrame = 0;

	std::string mFilePath = "test.bssrdf";

	int mSimulationSamplesPerFrame = (int)1e7;
	int mSimulationFrames = 2;
	int mSimulationMaxIterations = (int)1e4;

	float * mCurrentHemisphereData = nullptr;
	bool mPaused = false;
	RenderMode mCurrentRenderMode = RENDER_BSSRDF;
	std::unique_ptr<BSSRDFExporter> mExporter = nullptr;

	optix::Buffer mExternalBSSRDFBuffer;
	std::string mExternalFilePath = "test.bssrdf";

	void set_render_mode(RenderMode toapply);

};

