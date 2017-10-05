#pragma once
#include "reference_bssrdf.h"

class ReferenceBSSRDFGPU : public ReferenceBSSRDF
{
public:
	ReferenceBSSRDFGPU(const ShaderInfo& shader_info, int camera_width, int camera_height)
		: ReferenceBSSRDF(shader_info, camera_width, camera_height)
	{
	}

	void load_data() override;
	void initialize_shader(optix::Context) override;
	void initialize_mesh(Mesh& object) override;
	void pre_trace_mesh(Mesh& object) override;
	void post_trace_mesh(Mesh& object) override;
	bool on_draw() override;
	Shader* clone() override;
	~ReferenceBSSRDFGPU();
	
	optix::Buffer mAtomicPhotonCounterBuffer;
	optix::Buffer mPhotonBuffer;
	void reset() override;
	unsigned int mBatchIterations = (int)1e5;
};

