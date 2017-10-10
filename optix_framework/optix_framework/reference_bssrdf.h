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
	void load_data(Mesh & object) override;

	void set_geometry_parameters(float theta_i, float theta_s, float r)
	{
		mThetai = theta_i;
		mThetas = theta_s;
		mRadius = r;
	}

	void set_use_mesh_parameters(bool val)
	{
		mUseMeshParameters = val;
	}

	void set_material_parameters(float albedo, float extinction, float g, float eta);

protected:
	void init_output(const char * function);
	virtual void reset();

	// Geometric properties
	float mThetai = 60.0f;
	float mThetas = 0.0f;
	float mRadius = 0.8f;
	// Scattering properties
	float mAlbedo = 0.9f;
	float mExtinction = 1.0f;
	float mAsymmetry = 0.0f;
	float mIor = 1.3f;

	// Data for the simulation
	unsigned int mSamples = (int)1e5;
	optix::uint2 mHemisphereSize = optix::make_uint2(160, 40);
	int mCameraWidth;
	int mCameraHeight;
	int mRenderedFrames = 0;
	unsigned int mMaxIterations = (int)1e5;
	bool mUseMeshParameters = true;

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

