#include "sampled_bssrdf.h"
#include "mesh.h"
#include "immediate_gui.h"
#include "scattering_material.h"
#include "parameter_parser.h"
#include "string_utils.h"

inline float computeSamplingMfp(SamplingMfpType::Type e, const optix::float3& t)
{
	switch (e)
	{
		case SamplingMfpType::X: return t.x;
		case SamplingMfpType::Y: return t.y;
		case SamplingMfpType::Z: return t.z;
		case SamplingMfpType::MIN: return optix::fminf(t);
		case SamplingMfpType::MAX: return optix::fmaxf(t);
		case SamplingMfpType::MEAN: return optix::dot(t, optix::make_float3(0.333f));
		case SamplingMfpType::NotValidEnumItem:
		default:
			return 0;
	}
}

#define ENUM_SELECTOR(enum_type) \
bool enum_selector_gui_##enum_type(enum_type::Type & type, std::string name, std::string id = "") \
{ \
	std::string elements = ""; \
	enum_type::Type t = enum_type::first(); \
	do \
	{ \
		elements += prettify(enum_type::to_string(t)) + '\0'; \
		t = enum_type::next(t); \
	} while (t != enum_type::NotValidEnumItem); \
 \
	id = (id == "")? name : id; \
 \
	if (ImmediateGUIDraw::Combo((std::string(name) + std::string("##") + id).c_str(), (int*)&type, elements.c_str(), enum_type::count())) \
	{ \
		return true; \
	} \
	return false; \
}


ENUM_SELECTOR(BssrdfSamplingType)
ENUM_SELECTOR(BssrdfSamplePointOnTangentTechnique)
ENUM_SELECTOR(SamplingMfpType)


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
	mSamplingType = SamplingMfpType::to_enum(ConfigParameters::get_parameter<std::string>("bssrdf", "bssrdf_sampling_mfp", SamplingMfpType::to_string(mSamplingType), (std::string("Part of transport/s coeff. used for sampling. Values: ") + SamplingMfpType::get_full_string()).c_str()));

}

void SampledBSSRDF::initialize_mesh(Mesh& object)
{
    mCurrentShaderSource = get_current_shader_source();
    mReloadShader = true;
	Shader::initialize_mesh(object);
	BufPtr<BSSRDFSamplingProperties> bufptr(mPropertyBuffer->getId());
 	object.mMaterial["bssrdf_sampling_properties"]->setUserData(sizeof(BufPtr<BSSRDFSamplingProperties>), &bufptr);
    mBSSRDF->load(object.get_main_material()->get_data().relative_ior, object.get_main_material()->get_data().scattering_properties);

	ScatteringDipole::Type t = mBSSRDF->get_type();
	object.mMaterial["selected_bssrdf"]->setUserData(sizeof(ScatteringDipole::Type), &t);
}

void SampledBSSRDF::load_data(Mesh & object)
{
	if (mHasChanged)
	{
		const ScatteringMaterialProperties& scattering_properties = object.get_main_material()->get_data().scattering_properties;
		const float relior = object.get_main_material()->get_data().relative_ior;
		Logger::info << "Reloading shader" << std::endl;

		const optix::float3 imfp = mBSSRDF->get_sampling_inverse_mean_free_path(scattering_properties);
		properties->sampling_inverse_mean_free_path = computeSamplingMfp(mSamplingType, imfp);

		initialize_buffer<BSSRDFSamplingProperties>(mPropertyBuffer, *properties);

		context["samples_per_pixel"]->setUint(mSamples);
        mBSSRDF->load(relior, scattering_properties);
		ScatteringDipole::Type t = mBSSRDF->get_type();
		object.mMaterial["selected_bssrdf"]->setUserData(sizeof(ScatteringDipole::Type), &t);

		bool isNN = properties->sampling_tangent_plane_technique == BssrdfSamplePointOnTangentTechnique::NEURAL_NETWORK_IMPORTANCE_SAMPLING;
        if(isNN)
        {
            mNNSampler->load(relior, scattering_properties);
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
	static ScatteringDipole::Type dipole = mBSSRDF->get_type();
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


    if(enum_selector_gui_BssrdfSamplingType(properties->sampling_method, "Sampling Technique"))
    {
        mHasChanged = true;
    }

	std::vector<const char*> elems2{ "Use exponential distribution", "Use neural network" };


    if(enum_selector_gui_BssrdfSamplePointOnTangentTechnique(properties->sampling_tangent_plane_technique, "Tangent plane estimation"))
    {
        Logger::info << "Changing sampling tangent point to " << BssrdfSamplePointOnTangentTechnique::to_string(properties->sampling_tangent_plane_technique) << std::endl;
        mHasChanged = true;
        std::string new_shader = get_current_shader_source();
        mReloadShader = new_shader != mCurrentShaderSource;
        mCurrentShaderSource = new_shader;
    }

    if(properties->sampling_tangent_plane_technique == BssrdfSamplePointOnTangentTechnique::EXPONENTIAL_DISK)
    {
        mHasChanged |= enum_selector_gui_SamplingMfpType(mSamplingType, "Transport coefficient element");
    }

	if(properties->sampling_tangent_plane_technique == BssrdfSamplePointOnTangentTechnique::UNIFORM_DISK)
	{
		mHasChanged |= ImGui::InputFloat("Disc radius", &properties->R_max);
	}

	if (properties->sampling_method == BssrdfSamplingType::BSSRDF_SAMPLING_TANGENT_PLANE)
	{
        if(properties->sampling_tangent_plane_technique == BssrdfSamplePointOnTangentTechnique::NEURAL_NETWORK_IMPORTANCE_SAMPLING)
        {
            mHasChanged |= mNNSampler->on_draw();
        }
        mHasChanged |= ImmediateGUIDraw::InputFloat("Distance from surface", &properties->d_max);
        mHasChanged |= ImGui::InputFloat("Min no ni", &properties->dot_no_ni_min);
    }

	mHasChanged |= ImmediateGUIDraw::Checkbox("Jacobian", (bool*)&properties->use_jacobian);
	mHasChanged |= ImmediateGUIDraw::Checkbox("Exclude backfaces", (bool*)&properties->exclude_backfaces);
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

