#include "bssrdf_creator.h"
#include "immediate_gui.h"
#include "optix_utils.h"
#include "GL\glew.h"
#include "folders.h"
#include <sstream>

void EmpiricalBSSRDFCreator::set_geometry_parameters(float theta_i, float r, float theta_s)
{
	mThetai = theta_i;
	mThetas = theta_s;
	mRadius = r;
	reset();
}

void EmpiricalBSSRDFCreator::load_data()
{
	context["ref_frame_number"]->setUint(mRenderedFrames);
	context["reference_bssrdf_theta_i"]->setFloat(mThetai);
	context["reference_bssrdf_theta_s"]->setFloat(mThetas);
	context["reference_bssrdf_radius"]->setFloat(mRadius);
	context["reference_bssrdf_rel_ior"]->setFloat(mIor);
}

void EmpiricalBSSRDFCreator::set_material_parameters(float albedo, float extinction, float g, float eta)
{
	mAlbedo = albedo;
	mExtinction = extinction;
	mAsymmetry = g;
	mIor = eta;
	reset();
}

void EmpiricalBSSRDFCreator::reset()
{
	clear_buffer(mBSSRDFBuffer);
	clear_buffer(mBSSRDFBufferIntermediate);
	mRenderedFrames = 0;
	ScatteringMaterialProperties c;
	fill_scattering_parameters_alternative(c, 1, mIor, optix::make_float3(mAlbedo), optix::make_float3(mExtinction), optix::make_float3(mAsymmetry));
	ScatteringMaterialProperties* cc = reinterpret_cast<ScatteringMaterialProperties*>(mProperties->map());
	*cc = c;
	mProperties->unmap();
}

void EmpiricalBSSRDFCreator::init()
{
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

bool EmpiricalBSSRDFCreator::on_draw(bool show_material_params)
{
	bool changed = false;
	changed |= ImmediateGUIDraw::SliderFloat("Incoming light angle (deg.)", &mThetai, 0, 90);
	changed |= ImmediateGUIDraw::InputFloat("Radius", &mRadius);
	changed |= ImmediateGUIDraw::SliderFloat("Angle on plane", &mThetas, 0, 360);
	if (show_material_params)
	{
		changed |= ImmediateGUIDraw::InputFloat("Albedo##RefAlbedo", &mAlbedo);
		changed |= ImmediateGUIDraw::InputFloat("Extinction##RefExtinction", &mExtinction);
		changed |= ImmediateGUIDraw::InputFloat("G##RefAsymmetry", &mAsymmetry);
		changed |= ImmediateGUIDraw::InputFloat("Relative IOR##RefRelIOR", &mIor);
	}

	if (changed)
	{
		reset();

	}

	return changed;
}

void PlanarBSSRDF::init()
{
	EmpiricalBSSRDFCreator::init();
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
	const optix::int3 c = context->getPrintLaunchIndex();
	context->setPrintLaunchIndex(20, 20, -1);
	context->launch(entry_point, mHemisphereSize.x, mHemisphereSize.y);
	context->launch(entry_point_post, mHemisphereSize.x, mHemisphereSize.y);
	context->setPrintLaunchIndex(c.x, c.y, c.z);
	mRenderedFrames++;
}

bool PlanarBSSRDF::on_draw(bool show_material_params)
{
	if (EmpiricalBSSRDFCreator::on_draw(show_material_params))
		reset();
	return false;
}

void PlanarBSSRDF::load_data()
{
	EmpiricalBSSRDFCreator::load_data();
	ScatteringMaterialProperties* cc = reinterpret_cast<ScatteringMaterialProperties*>(mProperties->map());
	cc->selected_bssrdf = mScatteringDipole;
	mProperties->unmap();
}

void ReferenceBSSRDF::init()
{
	EmpiricalBSSRDFCreator::init();
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

void ReferenceBSSRDF::render()
{
	context->launch(entry_point, mSamples);
	context->launch(entry_point_post, mHemisphereSize.x, mHemisphereSize.y);
	mRenderedFrames++;
}

void ReferenceBSSRDF::load_data()
{
	EmpiricalBSSRDFCreator::load_data();
	context["maximum_iterations"]->setUint(mMaxIterations);
	context["reference_bssrdf_samples_per_frame"]->setUint(mSamples);

}

bool ReferenceBSSRDF::on_draw(bool show_material_params)
{
	std::stringstream ss;
	ss << "Rendered: " << mRenderedFrames << " frames, " << mRenderedFrames*mSamples << " samples" << std::endl;
	ImmediateGUIDraw::Text(ss.str().c_str());
	bool changed = EmpiricalBSSRDFCreator::on_draw(show_material_params);
	changed |= ImmediateGUIDraw::InputInt("Samples", (int*)&mSamples);
	changed |= ImmediateGUIDraw::InputInt("Maximum iterations", (int*)&mMaxIterations);
	if (changed)
		reset();
	return false;
}

void ReferenceBSSRDF::set_samples(int samples)
{
	mSamples = samples;
	reset();
}

void ReferenceBSSRDF::set_max_iterations(int max_iter)
{
	mMaxIterations = max_iter;
	reset();
}
