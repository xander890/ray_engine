#pragma once
#include <shader.h>
#include <mesh_host.h>
#include <string>

class BSSRDFRenderer;

class HemisphereBSSRDFShader : public Shader
{
public:
	HemisphereBSSRDFShader(const ShaderInfo& shader_info, std::unique_ptr<BSSRDFRenderer>& creator, int camera_width, int camera_height);

	void initialize_shader(optix::Context) override;
	void initialize_mesh(Object& object) override;
	void pre_trace_mesh(Object& object) override;
	void post_trace_mesh(Object& object) override;
	bool on_draw() override;
	Shader* clone() override { return new HemisphereBSSRDFShader(*this); }
	void load_data(Object & object) override;
	
	void set_use_mesh_parameters(bool val)
	{
		mUseMeshParameters = val;
	}

	HemisphereBSSRDFShader(HemisphereBSSRDFShader &);

protected:
	void init_output();
	virtual void reset();


	int mCameraWidth;
	int mCameraHeight;
	bool mUseMeshParameters = true;

	// Gui
	float mScaleMultiplier = 2 * 1000000.0f;
	int mShowFalseColors = 0;

	static int entry_point_output;
	optix::Buffer mBSSRDFBufferTexture;
	optix::TextureSampler mBSSRDFHemisphereTex;

	std::shared_ptr<BSSRDFRenderer> ref_impl;
};

