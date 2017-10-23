#pragma once
#include <string>
#include <host_device_common.h>
#include <scattering_properties.h>

class BSSRDFHemisphereRenderer
{
public:
	BSSRDFHemisphereRenderer(optix::Context & ctx, const optix::uint2 & hemisphere = optix::make_uint2(160, 40), const unsigned int samples = (int)1e5) : context(ctx), mHemisphereSize(hemisphere) {
		//mProperties.selected_bssrdf = ScatteringDipole::STANDARD_DIPOLE_BSSRDF;
	}

	virtual void load_data();
	void set_geometry_parameters(float theta_i, float r, float theta_s);
	void set_material_parameters(float albedo, float extinction, float g, float eta);

	virtual void reset();
	virtual void init();
	virtual void render() = 0;

	optix::Buffer get_output_buffer() { return mBSSRDFBuffer; }
	optix::uint2 get_hemisphere_size() { return mHemisphereSize; }
	size_t get_storage_size() { return mHemisphereSize.x*mHemisphereSize.y; }
	virtual size_t get_samples() { return 1; }

	virtual bool on_draw(bool show_material_params);

	void set_read_only(bool is_read_only) { mIsReadOnly = is_read_only;  }
	
protected:
	int entry_point = -1;
	int entry_point_post = -1;

	optix::uint2 mHemisphereSize;
	optix::Context context = nullptr;

	unsigned int mRenderedFrames = 0;

	optix::Buffer mBSSRDFBufferIntermediate = nullptr;
	optix::Buffer mBSSRDFBuffer = nullptr;

	// Geometric properties
	float mThetai = 70.0f;
	float mThetas = 60.0f;
	float mRadius = 1.0f;
	// Scattering properties
	optix::Buffer mProperties;

	float mAlbedo = 0.3f;
	float mExtinction = 1.0f;
	float mAsymmetry = 0.9f;
	float mIor = 1.4f;
	bool mIsReadOnly = false;
	bool mInitialized = false;
};

class BSSRDFHemisphereSimulated : public BSSRDFHemisphereRenderer
{
public:
	BSSRDFHemisphereSimulated(optix::Context & ctx, const optix::uint2 & hemisphere = optix::make_uint2(160, 40), const unsigned int samples = (int)1e8) : BSSRDFHemisphereRenderer(ctx, hemisphere), mSamples(samples)
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

class BSSRDFHemisphereModel : public BSSRDFHemisphereRenderer
{
public:
	BSSRDFHemisphereModel(optix::Context & ctx, const optix::uint2 & hemisphere = optix::make_uint2(160, 40), const ScatteringDipole::Type & dipole = ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF) : BSSRDFHemisphereRenderer(ctx, hemisphere)
	{
		mScatteringDipole = dipole;
	}

	void init() override;
	void render() override;
	bool on_draw(bool show_material_params) override;
	void load_data() override;
	void set_dipole(ScatteringDipole::Type dip) { mScatteringDipole = dip;  }
	ScatteringDipole::Type mScatteringDipole;
};