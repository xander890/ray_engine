#pragma once

#include <shader.h>
#include <mesh.h>

class ReferenceBSSRDF : public Shader
{
public:
	ReferenceBSSRDF(const ShaderInfo& shader_info, int camera_width, int camera_height) : Shader(shader_info),
	                                                                                      entry_point(0),
	                                                                                      entry_point_output(0),
	                                                                                      mCameraWidth(camera_width),
	                                                                                      mCameraHeight(camera_height)
	{
	}

	void initialize_shader(optix::Context) override;
	void initialize_mesh(Mesh& object) override;
	void pre_trace_mesh(Mesh& object) override;
	void post_trace_mesh(Mesh& object) override;
	bool on_draw() override;
	Shader* clone() override { return new ReferenceBSSRDF(*this); }
	void load_data() override;

protected:
	void init_output(const char * function);
	virtual void reset();

	// Geometric properties
	float mThetai = 60.0f;
	float mThetas = 0.0f;
	float mRadius = 0.8f;

	// Data for the simulation
	unsigned int mSamples = (int)1e5;
	optix::uint2 mHemisphereSize = optix::make_uint2(160, 40);
	int mCameraWidth;
	int mCameraHeight;
	int mRenderedFrames = 0;
	unsigned int mMaxIterations = (int)1e5;

	// Gui
	float mScaleMultiplier = 2 * 1000000.0f;
	int mShowFalseColors = 0;

	// OptiX stuff
	int entry_point;
	int entry_point_output;
	optix::Buffer mBSSRDFBuffer;
	optix::Buffer mBSSRDFBufferTexture;
	optix::TextureSampler mBSSRDFHemisphereTex;

};

