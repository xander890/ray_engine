#pragma once

#include <shader.h>
#include <mesh.h>

class ReferenceBSSRDF
{
public:
	ReferenceBSSRDF(optix::Context & ctx, const optix::uint2 & hemisphere = optix::make_uint2(160,40), const unsigned int samples = (int)1e5) : context(ctx), mHemisphereSize(hemisphere), mSamples(samples) {}

	void set_geometry_parameters(float theta_i, float theta_s, float r)
	{
		mThetai = theta_i;
		mThetas = theta_s;
		mRadius = r;
	}

	virtual void load_data();
	void set_material_parameters(float albedo, float extinction, float g, float eta);

	void set_samples(int samples);
	void set_max_iterations(int max_iter);
	// Data for the simulation

	virtual void reset();
	virtual void init();
	virtual void render();

	optix::Buffer get_output_buffer() { return mBSSRDFBuffer; }
	optix::uint2 get_hemisphere_size() { return mHemisphereSize; }

	virtual void on_draw(bool show_material_params);


protected:
	// OptiX stuff
	static int entry_point;
	static int entry_point_post;
	unsigned int mSamples;
	optix::uint2 mHemisphereSize;
	optix::Context context = nullptr;

	unsigned int mMaxIterations = (int)1e5;
	unsigned int mRenderedFrames = 0;
	optix::Buffer mBSSRDFBufferIntermediate = nullptr;
	optix::Buffer mBSSRDFBuffer = nullptr;
	// Geometric properties
	float mThetai = 60.0f;
	float mThetas = 0.0f;
	float mRadius = 0.8f;
	// Scattering properties
	float mAlbedo = 0.9f;
	float mExtinction = 1.0f;
	float mAsymmetry = 0.0f;
	float mIor = 1.3f;
};

class PlanarBSSRDF : public ReferenceBSSRDF
{

};

class HemisphereBSSRDFShader : public Shader
{
public:
	HemisphereBSSRDFShader(const ShaderInfo& shader_info, int camera_width, int camera_height) : Shader(shader_info),
	                                                                                      mCameraWidth(camera_width),
	                                                                                      mCameraHeight(camera_height)
	{
	}

	void initialize_shader(optix::Context) override;
	void initialize_mesh(Mesh& object) override;
	void pre_trace_mesh(Mesh& object) override;
	void post_trace_mesh(Mesh& object) override;
	bool on_draw() override;
	Shader* clone() override { return new HemisphereBSSRDFShader(*this); }
	void load_data(Mesh & object) override;
	
	void set_use_mesh_parameters(bool val)
	{
		mUseMeshParameters = val;
	}

	HemisphereBSSRDFShader::HemisphereBSSRDFShader(HemisphereBSSRDFShader &);

protected:
	void init_output(const char * function);
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

	std::unique_ptr<ReferenceBSSRDF> ref_impl;
};

