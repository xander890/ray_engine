#pragma once

#include <shader.h>
#include <mesh.h>

class BSSRDFCreator
{
public:
	BSSRDFCreator(optix::Context & ctx, const optix::uint2 & hemisphere = optix::make_uint2(160,40), const unsigned int samples = (int)1e5) : context(ctx), mHemisphereSize(hemisphere) {
		//mProperties.selected_bssrdf = ScatteringDipole::STANDARD_DIPOLE_BSSRDF;
	}

	void set_geometry_parameters(float theta_i, float theta_s, float r);

	virtual void load_data();
	void set_material_parameters(float albedo, float extinction, float g, float eta);

	virtual void reset();
	virtual void init();
	virtual void render() = 0;

	optix::Buffer get_output_buffer() { return mBSSRDFBuffer; }
	optix::uint2 get_hemisphere_size() { return mHemisphereSize; }

	virtual bool on_draw(bool show_material_params);

protected:
	int entry_point = -1;
	int entry_point_post = -1;

	optix::uint2 mHemisphereSize;
	optix::Context context = nullptr;

	unsigned int mRenderedFrames = 0;
	
	optix::Buffer mBSSRDFBufferIntermediate = nullptr;
	optix::Buffer mBSSRDFBuffer = nullptr;

	// Geometric properties
	float mThetai = 60.0f;
	float mThetas = 0.0f;
	float mRadius = 0.8f;
	// Scattering properties
	optix::Buffer mProperties;

	float mAlbedo = 0.9f;
	float mExtinction = 1.0f;
	float mAsymmetry = 0.0f;
	float mIor = 1.3f;
};

class ReferenceBSSRDF : public BSSRDFCreator
{
public:
	ReferenceBSSRDF(optix::Context & ctx, const optix::uint2 & hemisphere = optix::make_uint2(160, 40), const unsigned int samples = (int)1e5) : BSSRDFCreator(ctx, hemisphere), mSamples(samples) 
	{
		init();
	}

	void init() override;
	void render() override;
	void load_data() override;
	bool on_draw(bool show_material_params) override;
	void set_samples(int samples);
	void set_max_iterations(int max_iter);
protected:
	unsigned int mSamples;
	unsigned int mMaxIterations = (int)1e6;
};

class PlanarBSSRDF : public BSSRDFCreator
{
public:
	PlanarBSSRDF(optix::Context & ctx, const optix::uint2 & hemisphere = optix::make_uint2(160, 40), const ScatteringDipole::Type & dipole = ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF) : BSSRDFCreator(ctx, hemisphere) 
	{
		mScatteringDipole = dipole;
		init();
	}

	void init() override;
	void render() override;
	bool on_draw(bool show_material_params) override;
	void load_data() override;
	ScatteringDipole::Type mScatteringDipole;
};


class HemisphereBSSRDFShader : public Shader
{
public:
	HemisphereBSSRDFShader(const ShaderInfo& shader_info, std::unique_ptr<BSSRDFCreator>& creator, int camera_width, int camera_height);

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

	std::shared_ptr<BSSRDFCreator> ref_impl;
};

