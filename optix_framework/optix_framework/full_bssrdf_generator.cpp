#include "full_bssrdf_generator.h"
#include "optix_utils.h"
#include "immediate_gui.h"
#include <bssrdf_creator.h>
#include "parserstringhelpers.h"
#include <fstream>
#include "parameter_parser.h"
#include "folders.h"
#include "dialogs.h"
#include <sstream>
#include "reference_bssrdf_gpu.h"
#include "GL\glew.h"
#pragma warning(disable : 4996)

std::string gui_string(std::vector<float> & data)
{
	std::stringstream ss;
	ss.precision(2);
	ss << std::fixed;
	for (auto c : data) ss << c << " ";
	return ss.str();
}

FullBSSRDFGenerator::FullBSSRDFGenerator(const char * config ) : config_file(config)
{
}

FullBSSRDFGenerator::~FullBSSRDFGenerator()
{
}

void FullBSSRDFGenerator::initialize_scene(GLFWwindow * window, InitialCameraData & camera_data)
{
	//m_context->setPrintEnabled(true);
	m_context->setPrintBufferSize(200);
	m_context->setPrintLaunchIndex(0, 0, 0);
	ConfigParameters::init(config_file);
	Folders::init();
	m_context["scene_epsilon"]->setFloat(1e-3f);
	auto top_node = m_context->createGroup();
	auto accel = m_context->createAcceleration("Bvh");
	top_node->setAcceleration(accel);
	m_context["top_shadower"]->set(top_node);
	m_context["top_object"]->set(top_node);
	m_context["frame"]->setInt(0);

	m_context["debug_index"]->setUint(optix::make_uint2(0, 0));
	m_context["bad_color"]->setFloat(optix::make_float3(0.5, 0, 0));

	creator = std::make_unique<ReferenceBSSRDFGPU>(m_context, optix::make_uint2(160, 40), (int)10e5);
	creator->init();

	std::string ptx_path_output = get_path_ptx("render_bssrdf_hemisphere.cu");
	optix::Program ray_gen_program_output = m_context->createProgramFromPTXFile(ptx_path_output, "render_ref");

	if (entry_point_output == -1)
		entry_point_output = add_entry_point(m_context, ray_gen_program_output);

	result_buffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT);
	result_buffer->setFormat(RT_FORMAT_FLOAT4);
	result_buffer->setSize(1024, 1024);

	mBSSRDFBufferTexture = m_context->createBuffer(RT_BUFFER_INPUT);
	mBSSRDFBufferTexture->setFormat(RT_FORMAT_FLOAT);
	mBSSRDFBufferTexture->setSize(creator->get_hemisphere_size().x, creator->get_hemisphere_size().y);

	mBSSRDFHemisphereTex = m_context->createTextureSampler();
	mBSSRDFHemisphereTex->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	mBSSRDFHemisphereTex->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	mBSSRDFHemisphereTex->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
	mBSSRDFHemisphereTex->setBuffer(mBSSRDFBufferTexture);
	mBSSRDFHemisphereTex->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
	mBSSRDFHemisphereTex->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);

	int s = mBSSRDFHemisphereTex->getId();
	m_context["resulting_flux_tex"]->setUserData(sizeof(TexPtr), &(s));
	m_context["output_buffer"]->setBuffer(result_buffer);

	gui = std::make_unique<ImmediateGUI>(window);

	//FullBSSRDFParameters p;

	//int m = 0;
	//long long sum = 0;
	//for (int i = 0; i < p.eta.size(); i++)
	//	for (int i2 = 0; i2 < p.g.size(); i2++)
	//		for (int i3 = 0; i3 < p.albedo.size(); i3++)
	//			for (int i4 = 0; i4 < p.theta_s.size(); i4++)
	//				for (int i5 = 0; i5 < p.r.size(); i5++)
	//					for (int i6 = 0; i6 < p.theta_i.size(); i6++)
	//					{
	//						sum += (long long)p.flatten(i6, i5, i4, i3, i2, i);
	//						m = max(m, (int)p.flatten(i6, i5, i4, i3, i2, i));
	//					}
	//long long tot = (long)(p.eta.size()* p.g.size()*p.albedo.size()*p.theta_s.size()*p.r.size()*p.theta_i.size());
	//Logger::info << m << " " << tot - 1;
	//Logger::info << " " << sum << " " << tot*(tot-1)/2 << std::endl;

	if (mCurrentHemisphereData == nullptr)
	{
		mCurrentHemisphereData = new float[creator->get_storage_size()];
	}

	mExternalBSSRDFBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);

}

void normalize(float * data, size_t size)
{
	float max_elem = 0.0f;
	for (int i = 0; i < size; i++)
	{
		max_elem = fmaxf(max_elem, data[i]);
	}
	for (int i = 0; i < size; i++)
	{
		data[i] /= max_elem;
	}
}

void FullBSSRDFGenerator::trace(const RayGenCameraData & camera_data)
{
	static int frame = 0;

	if (!mPaused)
	{
		m_context["frame"]->setInt(frame);

		creator->load_data();

		if (mCurrentRenderMode == RenderMode::RENDER_BSSRDF)
		{
			creator->render();

			m_context["show_false_colors"]->setUint(mShowFalseColors);
			m_context["reference_scale_multiplier"]->setFloat(mScaleMultiplier);

			if (is_rendering)
			{
				int i = 0;
			}

			void* source = creator->get_output_buffer()->map();
			void* dest = mBSSRDFBufferTexture->map();

			memcpy(mCurrentHemisphereData, source, creator->get_hemisphere_size().x*creator->get_hemisphere_size().y * sizeof(float));
			memcpy(dest, mCurrentHemisphereData, creator->get_hemisphere_size().x*creator->get_hemisphere_size().y * sizeof(float));
			normalize((float*)dest, (int)creator->get_storage_size());
			creator->get_output_buffer()->unmap();
			mBSSRDFBufferTexture->unmap();
		}

		RTsize w, h;
		result_buffer->getSize(w, h);
		m_context->launch(entry_point_output, w, h);

		frame++;
		update_rendering();
	}
}

optix::Buffer FullBSSRDFGenerator::get_output_buffer()
{
	return result_buffer;
}

bool vector_gui(const std::string & name, std::vector<float> & vec, std::string & storage)
{	
	static char buf[256];
	storage.copy(buf, storage.size());
	buf[storage.size() + 1] = '\0';
	if (ImGui::InputText(name.c_str(), buf, storage.size()))
	{
		std::vector<float> c = tovalue<std::vector<float>>(std::string(buf));
		storage = gui_string(vec);
		vec.clear();
		vec.insert(vec.begin(), c.begin(), c.end());
		return true;
	}
	return false;
}


void FullBSSRDFGenerator::post_draw_callback()
{
	gui->start_draw();
	gui->start_window("Ray tracing demo", 20, 20, 500, 600);
	
	ImmediateGUIDraw::InputFloat("Reference scale multiplier", &mScaleMultiplier);

	ImmediateGUIDraw::Checkbox("Show false colors", (bool*)&mShowFalseColors); 
	ImmediateGUIDraw::SameLine();
	ImmediateGUIDraw::Checkbox("Pause", (bool*)&mPaused);

	const char * comboelements[2] = { "Render BSSRDF", "Show Existing BSSRDF" };
	if (ImmediateGUIDraw::Combo("Select Render mode", (int*)&mCurrentRenderMode, comboelements, 2, 2))
	{
		set_render_mode(mCurrentRenderMode);
	}

	if (mCurrentRenderMode == RENDER_BSSRDF)
	{
		creator->on_draw(true);
	}
	else
	{
		if (mLoader != nullptr)
		{
			static std::vector<size_t> index(6);
			for (int i = 0; i < 6; i++)
			{
				std::string s;
				for (int k = 0; k < mLoader->get_parameters().at(i).size(); k++)
				{
					s += std::to_string(mLoader->get_parameters().at(i)[k]) + '\0';
				}
				if (ImmediateGUIDraw::Combo(mParameters.parameter_names[i].c_str(), (int*)&index[i], s.c_str(), (int)mLoader->get_parameters().at(i).size()))
				{
					std::vector<size_t> dims;
					mLoader->get_dimensions(dims);
					float * data = (float*)mExternalBSSRDFBuffer->map();
					mLoader->load_hemisphere(data, { 0,0,0,0,0,0 });
					normalize(data, dims[phi_o_index] * dims[theta_o_index]);
					mExternalBSSRDFBuffer->unmap();
				}
			}
		}
	}

	if (ImmediateGUIDraw::CollapsingHeader("Load existing BSSRDF"))
	{
		ImmediateGUIDraw::InputText("Path##BSSRDFExtPath", &mExternalFilePath[0], mExternalFilePath.size(), ImGuiInputTextFlags_ReadOnly);

		std::string filePath;
		if (ImmediateGUIDraw::Button("Choose bssrdf path...##BSSRDFExtPathButton"))
		{
			std::string filePath;
			if (Dialogs::openFileDialog(filePath))
			{
				mExternalFilePath = filePath;
			}
		}

		if (ImmediateGUIDraw::Button("Load BSSRDF"))
		{
			set_external_bssrdf(mExternalFilePath);
			set_render_mode(RenderMode::SHOW_EXISTING_BSSRDF);
		}
	}

	if (ImmediateGUIDraw::CollapsingHeader("Parameter Range"))
	{
		for (int i = 0; i < mParameters.parameters.size(); i++)
		{
			if (pStrings[i] == "")
				pStrings[i] = gui_string(mParameters.parameters[i]);
			vector_gui(std::string("Simulated ") + mParameters.parameter_names[i], mParameters.parameters[i], pStrings[i]);
		}
	}

	if (ImmediateGUIDraw::CollapsingHeader("Simulation Parameters"))
	{
		ImmediateGUIDraw::InputText("Path##BSSRDFDestPath", &mFilePath[0], mFilePath.size(), ImGuiInputTextFlags_ReadOnly);

		std::string filePath;
		if (ImmediateGUIDraw::Button("Choose bssrdf path...##BSSRDFDestPathButton"))
		{
			std::string filePath;
			if (Dialogs::saveFileDialog(filePath))
			{
				mFilePath = filePath;
			}
		}
		ImmediateGUIDraw::InputInt("Simulation Samples per frame", &mSimulationSamplesPerFrame);
		ImmediateGUIDraw::InputInt("Simulation Frames", &mSimulationFrames);
		ImmediateGUIDraw::InputInt("Simulation Maximum Iterations", &mSimulationMaxIterations);
	}

	if (!is_rendering && ImmediateGUIDraw::Button("Start Simulation"))
	{
		start_rendering();
	}
	if (is_rendering && ImmediateGUIDraw::Button("End Simulation"))
	{
		end_rendering();
	}

	gui->end_window();
	gui->end_draw();
}

void FullBSSRDFGenerator::start_rendering()
{
	is_rendering = true;

	size_t total_size = mParameters.size() * sizeof(float) * creator->get_storage_size();

	std::vector<size_t> dims = mParameters.get_dimensions();
	dims.push_back(creator->get_hemisphere_size().x);
	dims.push_back(creator->get_hemisphere_size().y);

	mExporter = std::make_unique<BSSRDFExporter>(mFilePath, dims, mParameters.parameters, mParameters.parameter_names);
	
	mState = ParameterState({ 0,0,0,0,0,0 });
	creator->set_samples(mSimulationSamplesPerFrame);
	creator->set_max_iterations(mSimulationMaxIterations);
	mSimulationCurrentFrame = 0;
	creator->set_read_only(true);

	float extinction = 1.0f;
	float theta_i; float r; float theta_s; float albedo;  float g; float eta;
	mParameters.get_parameters(mState, theta_i, r, theta_s, albedo, g, eta);
	creator->set_geometry_parameters(theta_i, r, theta_s);
	creator->set_material_parameters(albedo, extinction, g, eta);
}

void FullBSSRDFGenerator::update_rendering()
{
	if (is_rendering)
	{
		if (mSimulationCurrentFrame == mSimulationFrames - 1)
		{
			Logger::info << "Simulation frame "<< mState.tostring() <<" complete. ("<< flatten_index(mState.mData, mParameters.get_dimensions()) <<"/"<< mParameters.get_size() <<") " << creator->get_samples() << " samples. " << std::endl;
			float extinction = 1.0f;
			float theta_i; float r; float theta_s; float albedo;  float g; float eta;
			mParameters.get_parameters(mState, theta_i, r, theta_s, albedo, g, eta);

			creator->set_geometry_parameters(theta_i, r, theta_s);
			creator->set_material_parameters(albedo, extinction, g, eta);

			mExporter->set_hemisphere(mCurrentHemisphereData, mState.mData);

			mState = mParameters.next(mState);
			mSimulationCurrentFrame = 0;

			if (!mParameters.is_valid(mState))
			{
				end_rendering();
			}

		}
		else
		{
			mSimulationCurrentFrame++;
		}
	}
}

void FullBSSRDFGenerator::end_rendering()
{
	is_rendering = false;
	creator->set_read_only(false);
}

bool FullBSSRDFGenerator::key_pressed(int key, int x, int y)
{
	return gui->keyPressed(key,x,y);
}

bool FullBSSRDFGenerator::mouse_pressed(int x, int y, int button, int action, int mods)
{
	return gui->mousePressed(x, y,button,action,mods);
}

bool FullBSSRDFGenerator::mouse_moving(int x, int y)
{
	return gui->mouseMoving(x, y);
}

void FullBSSRDFGenerator::clean_up()
{
	delete[] mCurrentHemisphereData;
}

void FullBSSRDFGenerator::set_external_bssrdf(const std::string & file)
{
	mLoader = std::make_unique<BSSRDFLoader>(file, mParameters.parameter_names);
	std::vector<size_t> dims;
	mLoader->get_dimensions(dims);
	mExternalBSSRDFBuffer->setSize(dims[phi_o_index], dims[theta_o_index]);
	float * data = (float*)mExternalBSSRDFBuffer->map();
	mLoader->load_hemisphere(data, {0,0,0,0,0,0});
	normalize(data, dims[phi_o_index] * dims[theta_o_index]);
	mExternalBSSRDFBuffer->unmap();
}

void FullBSSRDFGenerator::set_render_mode(RenderMode toapply)
{
	mCurrentRenderMode = toapply;
	if (mCurrentRenderMode == RenderMode::RENDER_BSSRDF)
	{
		mBSSRDFHemisphereTex->setBuffer(mBSSRDFBufferTexture);
	}
	else if (mCurrentRenderMode == RenderMode::SHOW_EXISTING_BSSRDF)
	{
		mBSSRDFHemisphereTex->setBuffer(mExternalBSSRDFBuffer);
	}
}

std::vector<size_t> FullBSSRDFParameters::get_dimensions()
{
	std::vector<size_t> dims(parameters.size());
	for (int i = 0; i < dims.size(); i++)
		dims[i] = parameters[i].size();
	return dims;
}
