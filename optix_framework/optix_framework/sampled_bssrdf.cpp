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
}

void SampledBSSRDF::initialize_shader(optix::Context ctx)
{
	Shader::initialize_shader(ctx);
	mPropertyBuffer = create_buffer<BSSRDFSamplingProperties>(context);
	properties->sampling_method = BssrdfSamplingType::to_enum(ConfigParameters::get_parameter<std::string>("bssrdf", "sampling_method", BssrdfSamplingType::to_string(properties->sampling_method), "Sampling method for illum 14. Available : " + BssrdfSamplingType::get_full_string()));
	Logger::info << "Using enum " << BssrdfSamplingType::to_string(properties->sampling_method) << std::endl;
	mBSSRDF = BSSRDF::create(context, ScatteringDipole::DIRECTIONAL_DIPOLE_BSSRDF);
}

void SampledBSSRDF::initialize_mesh(Mesh& object)
{
	Shader::initialize_mesh(object);
	BufPtr<BSSRDFSamplingProperties> bufptr(mPropertyBuffer->getId());
 	object.mMaterial["bssrdf_sampling_properties"]->setUserData(sizeof(BufPtr<BSSRDFSamplingProperties>), &bufptr);
	mBSSRDF->load(object.get_main_material()->get_data().scattering_properties);
	object.mMaterial["selected_bssrdf"]->setUserData(sizeof(ScatteringDipole::Type), &mBSSRDF->get_type());
}

void SampledBSSRDF::load_data(Mesh & object)
{
	if (mHasChanged)
	{
		Logger::info << "Reloading shader" << std::endl;
		initialize_buffer<BSSRDFSamplingProperties>(mPropertyBuffer, *properties);
		context["samples_per_pixel"]->setUint(mSamples);
		mBSSRDF->load(object.get_main_material()->get_data().scattering_properties);
		object.mMaterial["selected_bssrdf"]->setUserData(sizeof(ScatteringDipole::Type), &mBSSRDF->get_type());
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
	}
	ImmediateGUIDraw::Text("BSSRDF properties:");
	mBSSRDF->on_draw();


	std::vector<const char*> elems{ "Camera based (Mertens et. al)", "Tangent plane (with distance)" , "Tangent plane (with probes)", "MIS axis (no probes)",  "MIS axis + probes (King et al.)" };
	

	if (ImmediateGUIDraw::Combo("Sampling technique", (int*)&properties->sampling_method, elems.data(), (int)elems.size(), (int)elems.size()))
	{
		mHasChanged = true;
	}
	mHasChanged |= ImmediateGUIDraw::Checkbox("Jacobian", (bool*)&properties->use_jacobian);
	if (properties->sampling_method == BssrdfSamplingType::BSSRDF_SAMPLING_TANGENT_PLANE)
	{
		mHasChanged |= ImmediateGUIDraw::DragFloat("Distance from surface", &properties->d_max, 0.1f, 0.0f, 1.0f);
		mHasChanged |= ImGui::InputFloat("Min no ni", &properties->dot_no_ni_min);
	}

	mHasChanged |= ImGui::RadioButton("Show all", &properties->show_mode, BSSRDF_SHADERS_SHOW_ALL); ImGui::SameLine();
	mHasChanged |= ImGui::RadioButton("Refraction", &properties->show_mode, BSSRDF_SHADERS_SHOW_REFRACTION); ImGui::SameLine();
	mHasChanged |= ImGui::RadioButton("Reflection", &properties->show_mode, BSSRDF_SHADERS_SHOW_REFLECTION);

	mHasChanged |= ImGui::InputInt("Samples per pixel", (int*)&mSamples);

	return mHasChanged;
}

