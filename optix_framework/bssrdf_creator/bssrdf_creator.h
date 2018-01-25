#pragma once
#include <string>
#include <host_device_common.h>
#include <scattering_properties.h>
#include <full_bssrdf_host_device_common.h>
#include <bssrdf_properties.h>
#include <bssrdf_host.h>
#include "photon_trace_structs.h"

class BSSRDFRenderer
{
public:

	BSSRDFRenderer(optix::Context & ctx, const OutputShape::Type shape = OutputShape::HEMISPHERE, const optix::int2 & shape_size = optix::make_int2(-1)) : context(ctx) {
		mOutputShape = shape;
        if(shape_size.x > -1 && shape_size.y > -1)
        {
            mShapeSize = optix::make_uint2(shape_size.x, shape_size.y);
        }
        else
        {
            mShapeSize = default_size(shape);
        }
	}

	virtual void load_data();
	void set_geometry_parameters(float theta_i, optix::float2 r, optix::float2 theta_s);
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
	void set_shape(OutputShape::Type shape);
	OutputShape::Type get_shape() { return mOutputShape; }
	float mIor = 1.f;

protected:
	OutputShape::Type mOutputShape;
	int entry_point = -1;
	int entry_point_post = -1;

	optix::uint2 mShapeSize;
	optix::Context context = nullptr;

	unsigned int mRenderedFrames = 0;

	optix::Buffer mBSSRDFBufferIntermediate = nullptr;
	optix::Buffer mBSSRDFBuffer = nullptr;

	// Geometric properties

    BSSRDFRendererData mGeometryData;

	float mThetai = 0.0f;
	optix::float2 mThetas = optix::make_float2(0.0f, 7.5f);
    optix::float2 mRadius = optix::make_float2(0.0f, 1.0f);
	// Scattering properties
	optix::Buffer mProperties;

	float mAlbedo = 0.8f;
	float mExtinction = 1.0f;
	float mAsymmetry = 0.0f;
	bool mIsReadOnly = false;
	bool mInitialized = false;

    void fill_geometry_data();

	static optix::uint2 default_size(OutputShape::Type shape)
	{
		return shape == OutputShape::HEMISPHERE ? optix::make_uint2(160,40) : optix::make_uint2(400, 400);
	}
};

class BSSRDFRendererSimulated : public BSSRDFRenderer
{
public:
	BSSRDFRendererSimulated(optix::Context & ctx, const OutputShape::Type shape = OutputShape::HEMISPHERE, const optix::int2 & shape_size = optix::make_int2(-1), const unsigned int samples = (int)1e9) : BSSRDFRenderer(ctx, shape, shape_size), mSamples(samples)
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
	unsigned int mMaxIterations = (int)1e9;
	IntegrationMethod::Type mIntegrationMethod = IntegrationMethod::MCML;
    float mBiasCompensationBound = -1.0f;
};

class BSSRDFRendererModel : public BSSRDFRenderer
{
public:
	BSSRDFRendererModel(optix::Context & ctx, const OutputShape::Type shape = OutputShape::HEMISPHERE, const optix::int2 & shape_size = optix::make_int2(-1), const ScatteringDipole::Type & dipole = ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF) : BSSRDFRenderer(ctx, shape, shape_size)
	{
		mBSSRDF = std::move(BSSRDF::create(ctx, dipole));
	}

	void init() override;
	void render() override;
	bool on_draw(bool show_material_params) override;
	void load_data() override;
	void set_dipole(ScatteringDipole::Type dip);
	ScatteringDipole::Type get_dipole() {return mBSSRDF->get_type();}
	std::unique_ptr<BSSRDF> mBSSRDF = nullptr;
};