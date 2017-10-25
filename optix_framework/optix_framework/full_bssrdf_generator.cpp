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
#include "cputimer.h"
#include "GLFW\glfw3.h"
#include "GLFWDisplay.h"
#pragma warning(disable : 4996)



void get_default_material(const std::string & mat, float & theta_i, float & r, float & theta_s, float & albedo, float & extinction, float & g, float & n2_over_n1)
{
	if (mat == "A")
	{
		theta_i = 30.0f;
		albedo = 0.6f;
		extinction = 1.0f;
		g = 0.0f;
		n2_over_n1 = 1.3f;
		r = 4.0f;
		theta_s = 0;
	}
	else if (mat == "B")
	{
		theta_i = 60.0f;
		theta_s = 60;
		r = 0.8f;
		albedo = 0.99f;
		extinction = 1.0f;
		g = -0.3f;
		n2_over_n1 = 1.4f;
	}
	else if (mat == "C")
	{
		theta_i = 70.0f;
		theta_s = 60;
		r = 1.0f;
		albedo = 0.3f;
		extinction = 1.0f;
		g = 0.9f;
		n2_over_n1 = 1.4f;
	}
	else if (mat == "D")
	{
		theta_i = 0.0f;
		theta_s = 105.0f;
		r = 4.0f;
		albedo = 0.5f;
		extinction = 1.0f;
		g = 0.0f;
		n2_over_n1 = 1.2f;
	}
	else if (mat == "E")
	{
		theta_i = 80.0f;
		theta_s = 165.0f;
		r = 2.0f;
		albedo = 0.8f;
		extinction = 1.0f;
		g = -0.3f;
		n2_over_n1 = 1.3f;
	}
	else if (mat == "F")
	{
		theta_i = 80.0f;
		theta_s = 105.0f;
		r = .6f;
		albedo = 0.5f;
		extinction = 1.0f;
		g = -0.9f;
		n2_over_n1 = 1.4f;
	}
}

std::string gui_string(std::vector<float> & data)
{
	std::stringstream ss;
	ss.precision(2);
	ss << std::fixed;
	for (auto c : data) ss << c << " ";
	return ss.str();
}

FullBSSRDFGenerator::FullBSSRDFGenerator(const char * config , bool offline) : config_file(config), start_offline_rendering(offline)
{
	current_render_task = std::make_unique<RenderTaskTimeorFrames>(100, 10.0f, "test.bssrdf", false);
}

FullBSSRDFGenerator::~FullBSSRDFGenerator()
{
}

void FullBSSRDFGenerator::initialize_scene(GLFWwindow * window, InitialCameraData & camera_data)
{
	m_context->setPrintEnabled(false);
	m_context->setPrintBufferSize(200);
	m_context->setPrintLaunchIndex(0, 0, 0);
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

	mBssrdfReferenceSimulator = std::make_shared<ReferenceBSSRDFGPU>(m_context, BSSRDFRenderer::HEMISPHERE, optix::make_uint2(160, 40), (int)10e5);
	mBssrdfReferenceSimulator->init();

	mBssrdfModelSimulator = std::make_shared<BSSRDFRendererModel>(m_context);
	mBssrdfModelSimulator->init();

	set_render_mode(mCurrentRenderMode, mSimulate);

	std::string ptx_path_output = get_path_ptx("render_bssrdf.cu");
	optix::Program ray_gen_program_output = m_context->createProgramFromPTXFile(ptx_path_output, "render_ref");

	if (entry_point_output == -1)
		entry_point_output = add_entry_point(m_context, ray_gen_program_output);

	if (GLFWDisplay::isDisplayAvailable())
	{
		result_buffer = create_glbo_buffer < optix::float4 >(m_context, RT_BUFFER_INPUT_OUTPUT, 1024 * 1024);
	}
	else
	{
		result_buffer = create_buffer < optix::float4 >(m_context, RT_BUFFER_INPUT_OUTPUT, 1024 * 1024);
	}

	result_buffer->setFormat(RT_FORMAT_FLOAT4);
	result_buffer->setSize(1024, 1024);

	mBSSRDFBufferTexture = m_context->createBuffer(RT_BUFFER_INPUT);
	mBSSRDFBufferTexture->setFormat(RT_FORMAT_FLOAT);
	mBSSRDFBufferTexture->setSize(mCurrentBssrdfRenderer->get_size().x, mCurrentBssrdfRenderer->get_size().y);

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

	gui = nullptr;
	if (window != nullptr)
	{
		gui = std::make_unique<ImmediateGUI>(window);
	}

	if (mCurrentHemisphereData == nullptr)
	{
		mCurrentHemisphereData = new float[mCurrentBssrdfRenderer->get_storage_size()];
	}
	
	mParametersSimulation.parameters[eta_index] = ConfigParameters::get_parameter <std::vector<float>>("simulation", "eta", mParametersOriginal.parameters[eta_index], "Eta indices to scan.");
	mParametersSimulation.parameters[g_index] = ConfigParameters::get_parameter <std::vector<float>>("simulation", "g", mParametersOriginal.parameters[g_index], "Eta indices to scan.");
	mParametersSimulation.parameters[albedo_index] = ConfigParameters::get_parameter <std::vector<float>>("simulation", "albedo", mParametersOriginal.parameters[albedo_index], "Eta indices to scan.");
	
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
		double time = currentTime();

		m_context["frame"]->setInt(frame);

		mCurrentBssrdfRenderer->load_data();

		if (mCurrentRenderMode == RenderMode::RENDER_BSSRDF)
		{
			mCurrentBssrdfRenderer->render();

			m_context["show_false_colors"]->setUint(mShowFalseColors);
			m_context["reference_scale_multiplier"]->setFloat(mScaleMultiplier);

			void* source = mCurrentBssrdfRenderer->get_output_buffer()->map();

			memcpy(mCurrentHemisphereData, source, mCurrentBssrdfRenderer->get_size().x*mCurrentBssrdfRenderer->get_size().y * sizeof(float));
			if (!mFastMode)
			{
				void* dest = mBSSRDFBufferTexture->map();
				memcpy(dest, mCurrentHemisphereData, mCurrentBssrdfRenderer->get_size().x*mCurrentBssrdfRenderer->get_size().y * sizeof(float));

				if (mCurrentBssrdfRenderer->get_shape() == BSSRDFRenderer::HEMISPHERE)
					normalize((float*)dest, (int)mCurrentBssrdfRenderer->get_storage_size());
				mBSSRDFBufferTexture->unmap();
			}
			mCurrentBssrdfRenderer->get_output_buffer()->unmap();
		}

		double time1 = currentTime();

		if (!mFastMode)
		{
			RTsize w, h;
			result_buffer->getSize(w, h);
			m_context["ior"]->setFloat(mCurrentBssrdfRenderer->mIor);
			m_context->launch(entry_point_output, w, h);
		}

		frame++;

		update_rendering(static_cast<float>(time1 - time));
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
	buf[storage.size()] = '\0';

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
	if (gui == nullptr)
		return;
	gui->start_draw();
	gui->start_window("Ray tracing demo", 20, 20, 500, 600);
	
	ImmediateGUIDraw::InputFloat("Reference scale multiplier", &mScaleMultiplier);

	ImmediateGUIDraw::Checkbox("Show false colors", (bool*)&mShowFalseColors); 
	ImmediateGUIDraw::SameLine();
	ImmediateGUIDraw::Checkbox("Pause", (bool*)&mPaused);
	ImmediateGUIDraw::Checkbox("Fast Mode", (bool*)&mFastMode);

	const char * comboelements[2] = { "Render BSSRDF", "Show Existing Empirical BSSRDF" };
	if (ImmediateGUIDraw::Combo("Select Render mode", (int*)&mCurrentRenderMode, comboelements, 2, 2))
	{
		set_render_mode(mCurrentRenderMode, mSimulate);
	}

	if (mCurrentRenderMode == RENDER_BSSRDF)
	{
		const char * opts1[2] = { "BSSRDF model", "Path traced"};
		static int def1 = static_cast<int>(mSimulate);

		if (ImmediateGUIDraw::Combo("Simulation type", &def1 , opts1, 2, 2))
		{
			set_render_mode(mCurrentRenderMode, def1 == 0? false : true);
		}

		const char * opts2[2] = { "Plane", "Hemisphere" };
		static BSSRDFRenderer::OutputShape def2 = BSSRDFRenderer::HEMISPHERE;

		if (ImmediateGUIDraw::Combo("Render type", (int*)&def2, opts2, 2, 2))
		{
			mCurrentBssrdfRenderer->set_shape(def2);
			delete[] mCurrentHemisphereData;
			mCurrentHemisphereData = new float[mCurrentBssrdfRenderer->get_storage_size()];
			mBSSRDFBufferTexture->setSize(mCurrentBssrdfRenderer->get_size().x, mCurrentBssrdfRenderer->get_size().y);
		}

		if (!mSimulate)
		{
			std::string dipoles = "";
			ScatteringDipole::Type t = ScatteringDipole::first();
			do
			{
				dipoles += prettify(ScatteringDipole::to_string(t)) + '\0';
				t = ScatteringDipole::next(t);
			} while (t != ScatteringDipole::NotValidEnumItem);

			static ScatteringDipole::Type dipole;
			if (ImmediateGUIDraw::Combo("Dipole", (int*)&dipole, dipoles.c_str(), ScatteringDipole::count()))
			{
				std::dynamic_pointer_cast<BSSRDFRendererModel>(mCurrentBssrdfRenderer)->set_dipole(dipole);
			}
		}

		mCurrentBssrdfRenderer->on_draw(true);
		const char * opts[6] = {"A","B","C","D","E", "F"};
		static int def = 0;
		if (ImmediateGUIDraw::Combo("Default material", &def, opts, 6,6))
		{
			float theta_i; float r; float theta_s; float albedo; float extinction; float g; float n2_over_n1;
			get_default_material(opts[def], theta_i, r, theta_s, albedo, extinction, g, n2_over_n1);
			mCurrentBssrdfRenderer->set_geometry_parameters(theta_i, r, theta_s);
			mCurrentBssrdfRenderer->set_material_parameters(albedo, 1, g, n2_over_n1);
		}		
	}
	else if(mCurrentRenderMode == SHOW_EXISTING_BSSRDF)
	{
		if (mLoader != nullptr)
		{
			static ParameterState index({0,0,0,0,0,0});
			for (int i = 0; i < 6; i++)
			{
				std::string s;
				for (int k = 0; k < mLoader->get_parameters().at(i).size(); k++)
				{
					s += std::to_string(mLoader->get_parameters().at(i)[k]) + '\0';
				}
				if (ImmediateGUIDraw::Combo(mParametersOriginal.parameter_names[i].c_str(), (int*)&index.mData[i], s.c_str(), (int)mLoader->get_parameters().at(i).size()))
				{
					std::vector<size_t> dims;
					mLoader->get_dimensions(dims);
					float * data = (float*)mExternalBSSRDFBuffer->map();
					mLoader->load_hemisphere(data, index.mData);
					normalize(data, dims[phi_o_index] * dims[theta_o_index]);
					mExternalBSSRDFBuffer->unmap();
					

					float theta_i; float r; float theta_s; float albedo;  float g; float eta;
					mParametersSimulation.get_parameters(index, theta_i, r, theta_s, albedo, g, eta);
					mCurrentBssrdfRenderer->set_geometry_parameters(theta_i, r, theta_s);
					mCurrentBssrdfRenderer->set_material_parameters(albedo, 1, g, eta);
				}
			}
		}
	}

	if (mCurrentRenderMode == SHOW_EXISTING_BSSRDF && ImmediateGUIDraw::CollapsingHeader("Load existing BSSRDF"))
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
			set_render_mode(RenderMode::SHOW_EXISTING_BSSRDF, mSimulate);
		}
	}

	if (mCurrentRenderMode == RENDER_BSSRDF && ImmediateGUIDraw::CollapsingHeader("Parameter Range"))
	{
		static std::vector<char*> data(0);

		if (data.size() == 0)
		{
			for (int i = 0; i < mParametersSimulation.parameters.size(); i++)
			{
				data.push_back(new char[256]);
				std::string s = gui_string(mParametersSimulation.parameters[i]);
				s.copy(&data[i][0], s.size());
				data[i][s.size()] = '\0';
			}
		}

		for (int i = 0; i < mParametersSimulation.parameters.size(); i++)
		{
			if (ImGui::InputText((std::string("Simulated ") + mParametersSimulation.parameter_names[i]).c_str(), data[i], 256, ImGuiInputTextFlags_EnterReturnsTrue))
			{
				std::vector<float> c = tovalue<std::vector<float>>(std::string(data[i]));
				std::string s = gui_string(c);
				s.copy(&data[i][0], s.size());
				mParametersSimulation.parameters[i].clear();
				mParametersSimulation.parameters[i].insert(mParametersSimulation.parameters[i].begin(), c.begin(), c.end());
			}
		}

		if (ImmediateGUIDraw::Button("Reset parameters"))
		{
			mParametersSimulation.parameters = mParametersOriginal.parameters;
		}
	}

	if (mCurrentRenderMode == RENDER_BSSRDF && ImmediateGUIDraw::CollapsingHeader("Simulation Parameters"))
	{
		ImmediateGUIDraw::InputInt("Simulation Samples per frame", &mSimulationSamplesPerFrame);
		ImmediateGUIDraw::InputInt("Maximum Simulation Frames", &mSimulationMaxFrames);
		ImmediateGUIDraw::InputInt("Maximum Scattering evts./frame", &mSimulationMaxIterations);

		static int current_item = 2;
		const char * items[3] = { "Frame based", "Time based", "Time/Frame (first one to finish)" };
		if (!current_render_task->is_active())
		{
			if (ImmediateGUIDraw::Combo("Render task type", &current_item, items, 3, 3))
			{
				if (current_item == 0)
					current_render_task = std::make_unique<RenderTaskFrames>(100, current_render_task->get_destination_file(), false);
				else if(current_item == 1)
					current_render_task = std::make_unique<RenderTaskTime>(10.0f, current_render_task->get_destination_file(), false);
				else
					current_render_task = std::make_unique<RenderTaskTimeorFrames>(100, 10.0f, current_render_task->get_destination_file(), false);
			}
		}
		current_render_task->on_draw();
		if (current_render_task->is_active())
		{
			const float progress_total = flatten_index(mState.mData, mParametersSimulation.get_dimensions()) / (float)mParametersSimulation.get_size();
			const auto dims = mParametersSimulation.get_dimensions();
			const size_t material_size = dims[theta_i_index] * dims[theta_s_index] * dims[r_index];
			const float progress_bssrdf = (flatten_index(mState.mData, mParametersSimulation.get_dimensions()) % material_size) / (float)material_size;
			ImmediateGUIDraw::ProgressBar(progress_total, ImVec2(-1,0),"Simulation progress");
			ImmediateGUIDraw::ProgressBar(progress_bssrdf, ImVec2(-1, 0),"Current material progress");

			if (ImmediateGUIDraw::Button("End Simulation"))
			{
				end_rendering();
			}
		}
		else
		{
			if (ImmediateGUIDraw::Button("Start Simulation"))
			{
				start_rendering();
			}
		}

	}

	gui->end_window();
	gui->end_draw();
}

void FullBSSRDFGenerator::start_rendering()
{
	Logger::info << "Simulation started. " << std::endl;
	Logger::info << current_render_task->to_string() << std::endl;
	Logger::info << "Destination file: " << current_render_task->get_destination_file() << std::endl;
	Logger::info << "Eta:     " << tostring(mParametersSimulation.parameters[eta_index]) << std::endl;
	Logger::info << "G:       " << tostring(mParametersSimulation.parameters[g_index]) << std::endl;
	Logger::info << "Albedo:  " << tostring(mParametersSimulation.parameters[albedo_index]) << std::endl;
	Logger::info << "Theta s: " << tostring(mParametersSimulation.parameters[theta_s_index]) << std::endl;
	Logger::info << "R:       " << tostring(mParametersSimulation.parameters[r_index]) << std::endl;
	Logger::info << "Theta i: " << tostring(mParametersSimulation.parameters[theta_i_index]) << std::endl;
	current_render_task->start();

	std::vector<size_t> dims = mParametersOriginal.get_dimensions();
	dims.push_back(mCurrentBssrdfRenderer->get_size().x);
	dims.push_back(mCurrentBssrdfRenderer->get_size().y);

	mExporter = std::make_unique<BSSRDFExporter>(current_render_task->get_destination_file(), dims, mParametersOriginal.parameters, mParametersOriginal.parameter_names);
	
	mState = ParameterState({ 0,0,0,0,0,0 });

	if (mSimulate)
	{
		std::dynamic_pointer_cast<BSSRDFRendererSimulated>(mCurrentBssrdfRenderer)->set_samples(mSimulationSamplesPerFrame);
		std::dynamic_pointer_cast<BSSRDFRendererSimulated>(mCurrentBssrdfRenderer)->set_max_iterations(mSimulationMaxIterations);
	}	mCurrentBssrdfRenderer->set_read_only(true);

	float extinction = 1.0f;
	float theta_i; float r; float theta_s; float albedo;  float g; float eta;
	mParametersSimulation.get_parameters(mState, theta_i, r, theta_s, albedo, g, eta);
	mCurrentBssrdfRenderer->set_geometry_parameters(theta_i, r, theta_s);
	mCurrentBssrdfRenderer->set_material_parameters(albedo, extinction, g, eta);
	mFastMode = true;
}

void FullBSSRDFGenerator::update_rendering(float deltaTime)
{
	if (current_render_task->is_active())
	{
		if (current_render_task->is_finished())
		{
			Logger::info << "Simulation frame " << flatten_index(mState.mData, mParametersSimulation.get_dimensions()) + 1 << "/" << mParametersSimulation.get_size() << " complete. ";
			if (mSimulate)
			{
				Logger::info << std::dynamic_pointer_cast<BSSRDFRendererSimulated>(mCurrentBssrdfRenderer)->get_samples() << " samples. ";
			}
			float theta_i; float r; float theta_s; float albedo;  float g; float eta;
			mParametersSimulation.get_parameters(mState, theta_i, r, theta_s, albedo, g, eta);
			Logger::info <<  std::endl;
			
			ParameterState original_state = ParameterState({0,0,0,0,0,0});
			if (!mParametersOriginal.get_index(theta_i, r, theta_s, albedo, g, eta, original_state))
				Logger::error << "Index mismatch" << std::endl;

			Logger::info << "Index: " << mState.tostring() << "(simulation) "<< original_state.tostring() << "(original) Parameters: eta " << eta << " g " << g << " albedo " << albedo << std::endl;
			mExporter->set_hemisphere(mCurrentHemisphereData, mState.mData);
			mState = mParametersSimulation.next(mState);
			float extinction = 1.0f;
			mParametersSimulation.get_parameters(mState, theta_i, r, theta_s, albedo, g, eta);

			if (!mParametersSimulation.is_valid(mState))
			{
				end_rendering();
			}
			else
			{
				mCurrentBssrdfRenderer->set_geometry_parameters(theta_i, r, theta_s);
				mCurrentBssrdfRenderer->set_material_parameters(albedo, extinction, g, eta);
				current_render_task->start();
			}

		}
		else
		{
			current_render_task->update(deltaTime);
		}
	}
}

void FullBSSRDFGenerator::end_rendering()
{
	current_render_task->end();
	mCurrentBssrdfRenderer->set_read_only(false);
}

bool FullBSSRDFGenerator::key_pressed(int key, int x, int y)
{
	if (gui == nullptr)
		return false;
	return gui->keyPressed(key,x,y);
}

bool FullBSSRDFGenerator::mouse_pressed(int x, int y, int button, int action, int mods)
{
	if (gui == nullptr)
		return false;
	return gui->mousePressed(x, y,button,action,mods);
}

bool FullBSSRDFGenerator::mouse_moving(int x, int y)
{
	if (gui == nullptr)
		return false;
	return gui->mouseMoving(x, y);
}

void FullBSSRDFGenerator::clean_up()
{
	SampleScene::clean_up();
	delete[] mCurrentHemisphereData;
}

void FullBSSRDFGenerator::scene_initialized()
{
	if (start_offline_rendering)
		start_rendering();
}

void FullBSSRDFGenerator::set_external_bssrdf(const std::string & file)
{
	mLoader = std::make_unique<BSSRDFLoader>(file, mParametersOriginal.parameter_names);
	std::vector<size_t> dims;
	mLoader->get_dimensions(dims);
	mExternalBSSRDFBuffer->setSize(dims[phi_o_index], dims[theta_o_index]);
	float * data = (float*)mExternalBSSRDFBuffer->map();
	mLoader->load_hemisphere(data, {0,0,0,0,0,0});
	normalize(data, dims[phi_o_index] * dims[theta_o_index]);
	mExternalBSSRDFBuffer->unmap();
}


void FullBSSRDFGenerator::set_render_mode(RenderMode m, bool isSimulated)
{
	mSimulate = isSimulated;
	mCurrentRenderMode = m;

	if (mBSSRDFHemisphereTex.get() != nullptr)
	{
		if (mCurrentRenderMode == RenderMode::RENDER_BSSRDF && mBSSRDFBufferTexture.get() != nullptr)
		{
			mBSSRDFHemisphereTex->setBuffer(mBSSRDFBufferTexture);
		}
		else if (mCurrentRenderMode == RenderMode::SHOW_EXISTING_BSSRDF && mExternalBSSRDFBuffer.get() != nullptr)
		{
			mBSSRDFHemisphereTex->setBuffer(mExternalBSSRDFBuffer);
		}
	}

	if(m == RENDER_BSSRDF)
		mCurrentBssrdfRenderer = mSimulate? std::static_pointer_cast<BSSRDFRenderer>(mBssrdfReferenceSimulator) : std::static_pointer_cast<BSSRDFRenderer>(mBssrdfModelSimulator);
}

std::vector<size_t> FullBSSRDFParameters::get_dimensions()
{
	std::vector<size_t> dims(parameters.size());
	for (int i = 0; i < dims.size(); i++)
		dims[i] = parameters[i].size();
	return dims;
}

void FullBSSRDFGenerator::set_render_task(std::unique_ptr<RenderTask>& task)
{
	if (!current_render_task->is_active())
		current_render_task = std::move(task);
	else
		Logger::error << "Wait of end of current task before setting a new one." << std::endl;
}
