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

protected:
	void init_output(const char * function);
	int entry_point;
	int entry_point_output;
	optix::Buffer mBSSRDFBuffer;
	unsigned int mSamples = 10000;
	optix::uint2 mHemisphereSize = optix::make_uint2(360, 90) * 4;
	int mCameraWidth;
	int mCameraHeight;
	int mRenderedFrames = 0;

	float mScaleMultiplier = 2 * 1000000.0f;
	unsigned int mMaxIterations = (int)1e6;
	virtual void reset();
	int mShowFalseColors = 1;
};

