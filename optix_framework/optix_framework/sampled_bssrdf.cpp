#include "sampled_bssrdf.h"
#include "mesh.h"
#include "immediate_gui.h"
#include "scattering_material.h"
#include "parameter_parser.h"

SampledBSSRDF::SampledBSSRDF(const ShaderInfo& shader_info) : Shader(shader_info)
{
	properties = std::make_unique<BSSRDFSamplingProperties>();
}

SampledBSSRDF::SampledBSSRDF(const SampledBSSRDF & cp) : Shader(cp)
{
	mPropertyBuffer = create_buffer<BSSRDFSamplingProperties>(context);
	properties = std::make_unique<BSSRDFSamplingProperties>();
	*properties = *cp.properties;
	auto type = cp.mBSSRDF->get_type();
	mBSSRDF = BSSRDF::create(context, type);
    mNNSampler = std::make_unique<NeuralNetworkSampler>(context);
}

void SampledBSSRDF::initialize_shader(optix::Context ctx)
{
	Shader::initialize_shader(ctx);
	mPropertyBuffer = create_buffer<BSSRDFSamplingProperties>(context);
	properties->sampling_method = BssrdfSamplingType::to_enum(ConfigParameters::get_parameter<std::string>("bssrdf", "sampling_method", BssrdfSamplingType::to_string(properties->sampling_method), "Sampling method for illum 14. Available : " + BssrdfSamplingType::get_full_string()));
	properties->sampling_tangent_plane_technique = BssrdfSamplePointOnTangentTechnique::to_enum(ConfigParameters::get_parameter<std::string>("bssrdf", "sampling_tangent_plane_technique", BssrdfSamplePointOnTangentTechnique::to_string(properties->sampling_tangent_plane_technique), "Sampling method for illum 14. Available : " + BssrdfSamplePointOnTangentTechnique::get_full_string()));
	Logger::info << "Using enum " << BssrdfSamplingType::to_string(properties->sampling_method) << std::endl;
	Logger::info << "Using enum " << BssrdfSamplePointOnTangentTechnique::to_string(properties->sampling_tangent_plane_technique) << std::endl;
	auto s = ScatteringDipole::to_enum(ConfigParameters::get_parameter<std::string>("bssrdf", "bssrdf_model", ScatteringDipole::to_string(ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF), "Default dipole. Available : " + ScatteringDipole::get_full_string()));
	Logger::info << "Using dipole " << ScatteringDipole::to_string(s) << std::endl;
	mBSSRDF = BSSRDF::create(context, s);

}

void SampledBSSRDF::initialize_mesh(Mesh& object)
{
    mCurrentShaderSource = get_current_shader_source();
    mReloadShader = true;
	Shader::initialize_mesh(object);
	BufPtr<BSSRDFSamplingProperties> bufptr(mPropertyBuffer->getId());
 	object.mMaterial["bssrdf_sampling_properties"]->setUserData(sizeof(BufPtr<BSSRDFSamplingProperties>), &bufptr);
    mBSSRDF->load(object.get_main_material()->get_data().relative_ior, object.get_main_material()->get_data().scattering_properties);
	object.mMaterial["selected_bssrdf"]->setUserData(sizeof(ScatteringDipole::Type), &mBSSRDF->get_type());
}

void SampledBSSRDF::load_data(Mesh & object)
{
	if (mHasChanged)
	{
		Logger::info << "Reloading shader" << std::endl;
		initialize_buffer<BSSRDFSamplingProperties>(mPropertyBuffer, *properties);
		context["samples_per_pixel"]->setUint(mSamples);
        mBSSRDF->load(object.get_main_material()->get_data().relative_ior, object.get_main_material()->get_data().scattering_properties);
		object.mMaterial["selected_bssrdf"]->setUserData(sizeof(ScatteringDipole::Type), &mBSSRDF->get_type());

		bool isNN = properties->sampling_tangent_plane_technique == BssrdfSamplePointOnTangentTechnique::NEURAL_NETWORK_IMPORTANCE_SAMPLING;
        if(isNN)
        {
            mNNSampler->load(object.get_main_material()->get_data().relative_ior,object.get_main_material()->get_data().scattering_properties);
        }

        if(mReloadShader)
        {
            object.set_shader(mCurrentShaderSource);
            mReloadShader = false;
        }
	}
	mHasChanged = false;
}

bool SampledBSSRDF::on_draw()
{
	static ScatteringDipole::Type dipole = ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF;
	if (BSSRDF::dipole_selector_gui(dipole))
	{
		mHasChanged = true;
		mBSSRDF.reset();
		mBSSRDF = BSSRDF::create(context, dipole);
        std::string new_shader = get_current_shader_source();
        mReloadShader = new_shader != mCurrentShaderSource;
        mCurrentShaderSource = new_shader;
	}
	ImmediateGUIDraw::Text("BSSRDF properties:");
	mBSSRDF->on_draw();


	std::vector<const char*> elems{ "Camera based (Mertens et. al)", "Tangent plane (with distance)" , "Tangent plane (with probes)", "MIS axis (King et al.)" };

	if (ImmediateGUIDraw::Combo("Sampling technique", (int*)&properties->sampling_method, elems.data(), (int)elems.size(), (int)elems.size()))
	{
		mHasChanged = true;
	}

	std::vector<const char*> elems2{ "Use exponential distribution", "Use neural network" };


	if (properties->sampling_method == BssrdfSamplingType::BSSRDF_SAMPLING_TANGENT_PLANE)
	{
		if(ImmediateGUIDraw::Combo("Estimate point on tangent",  (int*)&properties->sampling_tangent_plane_technique, elems2.data(), (int)elems2.size(), (int)elems2.size()))
        {
            bool isNN = properties->sampling_tangent_plane_technique == BssrdfSamplePointOnTangentTechnique::NEURAL_NETWORK_IMPORTANCE_SAMPLING;
            Logger::info << "Changing sampling tangent point to " << BssrdfSamplePointOnTangentTechnique::to_string(properties->sampling_tangent_plane_technique) << std::endl;
            mHasChanged = true;
            std::string new_shader = get_current_shader_source();
            mReloadShader = new_shader != mCurrentShaderSource;
            mCurrentShaderSource = new_shader;
        }

        if(properties->sampling_tangent_plane_technique == BssrdfSamplePointOnTangentTechnique::NEURAL_NETWORK_IMPORTANCE_SAMPLING)
        {
            mHasChanged |= mNNSampler->on_draw();
        }
	}

	mHasChanged |= ImmediateGUIDraw::Checkbox("Jacobian", (bool*)&properties->use_jacobian);
	if (properties->sampling_method == BssrdfSamplingType::BSSRDF_SAMPLING_TANGENT_PLANE)
	{
		mHasChanged |= ImmediateGUIDraw::InputFloat("Distance from surface", &properties->d_max);
		mHasChanged |= ImGui::InputFloat("Min no ni", &properties->dot_no_ni_min);
	}

	mHasChanged |= ImGui::RadioButton("Show all", &properties->show_mode, BSSRDF_SHADERS_SHOW_ALL); ImGui::SameLine();
	mHasChanged |= ImGui::RadioButton("Refraction", &properties->show_mode, BSSRDF_SHADERS_SHOW_REFRACTION); ImGui::SameLine();
	mHasChanged |= ImGui::RadioButton("Reflection", &properties->show_mode, BSSRDF_SHADERS_SHOW_REFLECTION);

	mHasChanged |= ImGui::InputInt("Samples per pixel", (int*)&mSamples);

	return mHasChanged;
}

std::string SampledBSSRDF::get_current_shader_source() {
    bool isNN = properties->sampling_tangent_plane_technique == BssrdfSamplePointOnTangentTechnique::NEURAL_NETWORK_IMPORTANCE_SAMPLING;
    std::string ret = mBSSRDF->get_type() == ScatteringDipole::FORWARD_SCATTERING_DIPOLE_BSSRDF? "subsurface_scattering_sampled_forward_dipole.cu" : "subsurface_scattering_sampled_default.cu";
    return isNN? "subsurface_scattering_sampled_neural_network.cu" : ret;
}

