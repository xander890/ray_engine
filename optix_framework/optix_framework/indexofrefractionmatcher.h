#pragma once
#include <vector>
#include <optix_world.h>

class ObjScene;

static const int MAX_CORNERS = 24 * 4;



class IndexMatcher
{
public:
	IndexMatcher(optix::Context & context, ObjScene * scene, int w = -1, int h = -1);

	~IndexMatcher() {}

	void init(int termination_tries, float termination_epsilon, float eta_starting_guess);
	
	void run();


private:
	std::vector<optix::float2> reference_uv_map;
	std::vector<optix::float4> reference_3d_map; //Map from screen to 3d coordinates

	std::vector<optix::float3> reference_3d_points; //Extracted 3d points for each reference uv.

	optix::Context & context;
	int frame = 0;
	bool first_run_performed = false;
	int termination_tries = 20;
	float termination_epsilon = 10e-4f;

	int entry_point;
	
	float eta = 1.0f;
	float cost = FLT_MAX;
	float last_frame_cost = 0.0f;
	ObjScene * scene;
	int width, height;

	void find_uv_correspondence(std::vector<optix::float4> & uv_buffer, std::vector<optix::int2> &correspondent_pixels, std::vector<optix::float3> & result);

	std::vector<optix::float2> costs;

	static const int UV_SIZE = 68;
	static const optix::float2 reference_corner_uvs[UV_SIZE];
	static const int CHECKERBOARD_UVS = 96;
	static const optix::float2 checkerboard_uvs[CHECKERBOARD_UVS];

};

