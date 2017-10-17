#include "bssrdf_creator.h"
#include "immediate_gui.h"
#include "optix_utils.h"
#include "GL\glew.h"
#include "folders.h"
#include <sstream>

void BSSRDFHemisphereRenderer::set_geometry_parameters(float theta_i, float r, float theta_s)
{
	mThetai = theta_i;
	mThetas = theta_s;
	mRadius = r;
	reset();
}

void BSSRDFHemisphereRenderer::load_data()
{
	if (!mInitialized)
		init();
	context["ref_frame_number"]->setUint(mRenderedFrames);
	context["reference_bssrdf_theta_i"]->setFloat(deg2rad(mThetai));
	context["reference_bssrdf_theta_s"]->setFloat(deg2rad(mThetas));
	context["reference_bssrdf_radius"]->setFloat(mRadius);
	context["reference_bssrdf_rel_ior"]->setFloat(mIor);
}

void BSSRDFHemisphereRenderer::set_material_parameters(float albedo, float extinction, float g, float eta)
{
	mAlbedo = albedo;
	mExtinction = extinction;
	mAsymmetry = g;
	mIor = eta;
	reset();
}

void BSSRDFHemisphereRenderer::reset()
{
	if (!mInitialized)
		init();
	clear_buffer(mBSSRDFBuffer);
	clear_buffer(mBSSRDFBufferIntermediate);
	mRenderedFrames = 0;
	ScatteringMaterialProperties c;
	fill_scattering_parameters_alternative(c, 1, mIor, optix::make_float3(mAlbedo), optix::make_float3(mExtinction), optix::make_float3(mAsymmetry));
	ScatteringMaterialProperties* cc = reinterpret_cast<ScatteringMaterialProperties*>(mProperties->map());
	*cc = c;
	mProperties->unmap();
}

void BSSRDFHemisphereRenderer::init()
{
	mInitialized = true;
	if (mBSSRDFBufferIntermediate.get() == nullptr)
	{
		mBSSRDFBufferIntermediate = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
		mBSSRDFBufferIntermediate->setFormat(RT_FORMAT_FLOAT);
		mBSSRDFBufferIntermediate->setSize(mHemisphereSize.x, mHemisphereSize.y);
	}

	if (mBSSRDFBuffer.get() == nullptr)
	{
		mBSSRDFBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
		mBSSRDFBuffer->setFormat(RT_FORMAT_FLOAT);
		mBSSRDFBuffer->setSize(mHemisphereSize.x, mHemisphereSize.y);
	}

	if (mProperties.get() == nullptr)
	{
		mProperties = context->createBuffer(RT_BUFFER_INPUT);
		mProperties->setFormat(RT_FORMAT_USER);
		mProperties->setElementSize(sizeof(ScatteringMaterialProperties));
		mProperties->setSize(1);
	}

	reset();
}

bool BSSRDFHemisphereRenderer::on_draw(bool show_material_params)
{
	bool changed = false;

	float backup_theta_i = mThetai;
	float backup_theta_s = mThetas;

	changed |= ImmediateGUIDraw::SliderFloat("Incoming light angle (deg.)", &mThetai, 0, 90);
	changed |= ImmediateGUIDraw::InputFloat("Radius", &mRadius, 0, 0, -1, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0);
	changed |= ImmediateGUIDraw::SliderFloat("Angle on plane", &mThetas, 0, 360);

	mThetai = mIsReadOnly ? backup_theta_i : mThetai;
	mThetas = mIsReadOnly ? backup_theta_s : mThetas;

	if (show_material_params)
	{
		changed |= ImmediateGUIDraw::InputFloat("Albedo##RefAlbedo", &mAlbedo, 0, 0, -1, mIsReadOnly? ImGuiInputTextFlags_ReadOnly : 0);
		changed |= ImmediateGUIDraw::InputFloat("Extinction##RefExtinction", &mExtinction, 0, 0, -1, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0);
		changed |= ImmediateGUIDraw::InputFloat("G##RefAsymmetry", &mAsymmetry, 0, 0, -1, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0);
		changed |= ImmediateGUIDraw::InputFloat("Relative IOR##RefRelIOR", &mIor, 0, 0, -1, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0);
	}

	if (changed)
	{
		reset();

	}

	return changed;
}

void PlanarBSSRDF::init()
{
	BSSRDFHemisphereRenderer::init();
	std::string ptx_path = get_path_ptx("planar_bssrdf.cu");
	optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_camera");
	optix::Program ray_gen_program_post = context->createProgramFromPTXFile(ptx_path, "post_process_bssrdf");

	if (entry_point == -1)
		entry_point = add_entry_point(context, ray_gen_program);
	if (entry_point_post == -1)
		entry_point_post = add_entry_point(context, ray_gen_program_post);

	BufPtr<ScatteringMaterialProperties> id = BufPtr<ScatteringMaterialProperties>(mProperties->getId());
	context["planar_bssrdf_material_params"]->setUserData(sizeof(BufPtr<ScatteringMaterialProperties>), &id);
	BufPtr2D<float> ptr = BufPtr2D<float>(mBSSRDFBufferIntermediate->getId());
	context["planar_resulting_flux_intermediate"]->setUserData(sizeof(BufPtr2D<float>), &ptr);
	BufPtr2D<float> ptr2 = BufPtr2D<float>(mBSSRDFBuffer->getId());
	context["planar_resulting_flux"]->setUserData(sizeof(BufPtr2D<float>), &ptr2);

}

void PlanarBSSRDF::render()
{
	if (!mInitialized)
		init();
	const optix::int3 c = context->getPrintLaunchIndex();
	context->setPrintLaunchIndex(20, 20, -1);
	context->launch(entry_point, mHemisphereSize.x, mHemisphereSize.y);
	context->launch(entry_point_post, mHemisphereSize.x, mHemisphereSize.y);
	context->setPrintLaunchIndex(c.x, c.y, c.z);
	mRenderedFrames++;
}

bool PlanarBSSRDF::on_draw(bool show_material_params)
{
	if (BSSRDFHemisphereRenderer::on_draw(show_material_params))
		reset();
	return false;
}

void PlanarBSSRDF::load_data()
{
	BSSRDFHemisphereRenderer::load_data();
	ScatteringMaterialProperties* cc = reinterpret_cast<ScatteringMaterialProperties*>(mProperties->map());
	cc->selected_bssrdf = mScatteringDipole;
	mProperties->unmap();
}

void BSSRDFHemisphereSimulated::init()
{
	BSSRDFHemisphereRenderer::init();
	std::string ptx_path = get_path_ptx("reference_bssrdf.cu");
	optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "reference_bssrdf_camera");
	optix::Program ray_gen_program_post = context->createProgramFromPTXFile(ptx_path, "post_process_bssrdf");

	if (entry_point == -1)
		entry_point = add_entry_point(context, ray_gen_program);
	if (entry_point_post == -1)
		entry_point_post = add_entry_point(context, ray_gen_program_post);

	BufPtr<ScatteringMaterialProperties> id = BufPtr<ScatteringMaterialProperties>(mProperties->getId());
	context["reference_bssrdf_material_params"]->setUserData(sizeof(BufPtr<ScatteringMaterialProperties>), &id);
	BufPtr2D<float> ptr = BufPtr2D<float>(mBSSRDFBufferIntermediate->getId());
	context["reference_resulting_flux_intermediate"]->setUserData(sizeof(BufPtr2D<float>), &ptr);
	BufPtr2D<float> ptr2 = BufPtr2D<float>(mBSSRDFBuffer->getId());
	context["reference_resulting_flux"]->setUserData(sizeof(BufPtr2D<float>), &ptr2);


}

void BSSRDFHemisphereSimulated::render()
{
	if (!mInitialized)
		init();
	context->launch(entry_point, mSamples);
	context->launch(entry_point_post, mHemisphereSize.x, mHemisphereSize.y);
	mRenderedFrames++;
}

void BSSRDFHemisphereSimulated::load_data()
{
	BSSRDFHemisphereRenderer::load_data();
	context["maximum_iterations"]->setUint(mMaxIterations);
	context["reference_bssrdf_samples_per_frame"]->setUint(mSamples);

}

bool BSSRDFHemisphereSimulated::on_draw(bool show_material_params)
{
	std::stringstream ss;
	ss << "Rendered: " << mRenderedFrames << " frames, " << mRenderedFrames*mSamples << " samples" << std::endl;
	ImmediateGUIDraw::Text(ss.str().c_str());
	bool changed = BSSRDFHemisphereRenderer::on_draw(show_material_params);

	static int smpl = mSamples;
	if (ImmediateGUIDraw::InputInt("Samples", (int*)&smpl, 1, 100, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0))
	{
		set_samples(smpl);
	}

	static int iter = mMaxIterations;
	if (ImmediateGUIDraw::InputInt("Maximum iterations", (int*)&iter, 1, 100, mIsReadOnly ? ImGuiInputTextFlags_ReadOnly : 0))
	{
		set_max_iterations(iter);
	}

	if (changed)
		reset();
	return false;
}

void BSSRDFHemisphereSimulated::set_samples(int samples)
{
	mSamples = samples;
	reset();
}

void BSSRDFHemisphereSimulated::set_max_iterations(int max_iter)
{
	mMaxIterations = max_iter;
	reset();
}
