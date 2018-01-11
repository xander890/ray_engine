#pragma once
#include <string>
#include <host_device_common.h>
#include <scattering_properties.h>
#include <full_bssrdf_host_device_common.h>
#include <bssrdf_properties.h>
#include <bssrdf_host.h>

class BSSRDFRenderer
{
public:
	enum OutputShape { PLANE = BSSRDF_OUTPUT_PLANE, HEMISPHERE = BSSRDF_OUTPUT_HEMISPHERE};

	BSSRDFRenderer(optix::Context & ctx, const OutputShape shape = HEMISPHERE, const optix::uint2 shape_size  = optix::make_uint2(0,0)) : context(ctx), mShapeSize(shape_size) {
		mOutputShape = shape;
		mShapeSize = default_size(shape);
	}

	virtual void load_data();
	void set_geometry_parameters(float theta_i, float r, float theta_s);
	void set_material_parameters(float albedo, float extinction, float g, float eta);

	virtual void reset();
	virtual void init();
	virtual void render() = 0;

	optix::Buffer get_output_buffer() { return mBSSRDFBuffer; }
	optix::uint2 get_size() { return mShapeSize; }
	size_t get_storage_size() { return mShapeSize.x*mShapeSize.y; }
	virtual size_t get_samples() { return 1; }

	virtual bool on_draw(bool show_material_params);

	void set_read_only(bool is_read_only) { mIsReadOnly = is_read_only;  }
	void set_shape(OutputShape shape);
	OutputShape get_shape() { return mOutputShape; }
	float mIor = 1.4f;

protected:
	OutputShape mOutputShape; 
	int entry_point = -1;
	int entry_point_post = -1;

	optix::uint2 mShapeSize;
	optix::Context context = nullptr;

	unsigned int mRenderedFrames = 0;

	optix::Buffer mBSSRDFBufferIntermediate = nullptr;
	optix::Buffer mBSSRDFBuffer = nullptr;

	// Geometric properties
	float mThetai = 0.0f;
	float mThetas = 0.0f;
	float mRadius = 1.0f;
	// Scattering properties
	optix::Buffer mProperties;

	float mAlbedo = 0.8f;
	float mExtinction = 1.0f;
	float mAsymmetry = 0.9f;
	bool mIsReadOnly = false;
	bool mInitialized = false;

private:
	static optix::uint2 default_size(OutputShape shape)
	{
		return shape == HEMISPHERE ? optix::make_uint2(160, 40) : optix::make_uint2(400, 400);
	}
};

class BSSRDFRendererSimulated : public BSSRDFRenderer
{
public:
	BSSRDFRendererSimulated(optix::Context & ctx, const OutputShape shape = HEMISPHERE, const optix::uint2 & shape_size = optix::make_uint2(160, 40), const unsigned int samples = (int)1e8) : BSSRDFRenderer(ctx, shape, shape_size), mSamples(samples)
	{
	}

	void init() override;
	void render() override;
	void load_data() override;
	bool on_draw(bool show_material_params) override;
	virtual void set_samples(int samples);
	virtual void set_max_iterations(int max_iter);
	size_t get_samples() override { return mSamples * mRenderedFrames; }

protected:
	unsigned int mSamples;
	unsigned int mMaxIterations = (int)1e4;

};

class BSSRDFRendererModel : public BSSRDFRenderer
{
public:
	BSSRDFRendererModel(optix::Context & ctx, const OutputShape shape = HEMISPHERE, const optix::uint2 & shape_size = optix::make_uint2(160, 40), const ScatteringDipole::Type & dipole = ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF) : BSSRDFRenderer(ctx, shape, shape_size)
	{
		mBSSRDF = std::move(BSSRDF::create(ctx, dipole));
	}

	void init() override;
	void render() override;
	bool on_draw(bool show_material_params) override;
	void load_data() override;
	void set_dipole(ScatteringDipole::Type dip);
	std::unique_ptr<BSSRDF> mBSSRDF = nullptr;
};