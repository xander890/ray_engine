#include "full_bssrdf_generator.h"
#include "optix_host_utils.h"
#include "immediate_gui.h"
#include "file_dialogs.h"
#include <fstream>
#include <camera_host.h>
#include <parsing_utils.h>
#include "cpu_timer.h"
#include "GLFW/glfw3.h"
#include "glfw_display.h"
#include "forward_dipole_test.h"
#include "reference_bssrdf_gpu_mixed.h"

#pragma warning(disable : 4996)

#define WINDOW_SIZE 800

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

FullBSSRDFGenerator::FullBSSRDFGenerator(bool offline) : start_offline_rendering(offline), mSimulationState(ParameterStateNew::invalid_index)
{
	current_render_task = std::make_unique<RenderTaskTimeorFrames>(100, 10.0f, "test.bssrdf", false);
}

FullBSSRDFGenerator::~FullBSSRDFGenerator()
{
}

void FullBSSRDFGenerator::initialize_scene(GLFWwindow * window)
{
	m_context->setPrintEnabled(mDebug);
// 	test_forward_dipole();
	m_context->setPrintBufferSize(2000);
	m_context->setPrintLaunchIndex(10000);
	m_context["scene_epsilon"]->setFloat(1e-3f);
	auto top_node = m_context->createGroup();
	auto accel = m_context->createAcceleration("Bvh");
	top_node->setAcceleration(accel);
	m_context["top_shadower"]->set(top_node);
	m_context["top_object"]->set(top_node);
	m_context["frame"]->setUint(0);

	m_context["debug_index"]->setUint(optix::make_uint2(0, 0));
	m_context["bad_color"]->setFloat(optix::make_float3(0.5, 0, 0));

	auto c = std::make_shared<ReferenceBSSRDFGPUMixed>(m_context, DEFAULT_SHAPE, optix::make_int2(160, 40), (int)10e5);
    mBssrdfReferenceSimulator = c;
	mBssrdfReferenceSimulator->init();
    c->set_preset(ReferenceRendererPreset::MIXED_BIAS_REDUCTION);

	mBssrdfModelSimulator = std::make_shared<BSSRDFRendererModel>(m_context);
	mBssrdfModelSimulator->set_dipole(ScatteringDipole::FORWARD_SCATTERING_DIPOLE_BSSRDF);
	mBssrdfModelSimulator->init();

	set_render_mode(mCurrentRenderMode, mSimulate);

	std::string ptx_path_output = Folders::get_path_to_ptx("render_bssrdf.cu");
	optix::Program ray_gen_program_output = m_context->createProgramFromPTXFile(ptx_path_output, "render_ref");

	if (entry_point_output == -1)
		entry_point_output = add_entry_point(m_context, ray_gen_program_output);

	if (GLFWDisplay::is_display_available())
	{
		result_buffer = create_glbo_buffer<optix::float4>(m_context, RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, WINDOW_SIZE * WINDOW_SIZE);
	}
	else
	{
		result_buffer = create_buffer < optix::float4 >(m_context, RT_BUFFER_INPUT_OUTPUT, WINDOW_SIZE * WINDOW_SIZE);
	}

	result_buffer->setFormat(RT_FORMAT_FLOAT4);
	result_buffer->setSize(WINDOW_SIZE, WINDOW_SIZE);

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
		gui = std::make_unique<ImmediateGUI>();
	}

	if (mCurrentHemisphereData == nullptr)
	{
		mCurrentHemisphereData = new float[mCurrentBssrdfRenderer->get_storage_size()];
	}

    mSimulationBSSRDF = std::make_unique<EmpiricalBSSRDF>();

    load_parameters("simulation_parameters.xml");

	mExternalBSSRDFBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, 1);


    if (!exists("simulation_parameters.xml"))
        save_parameters("simulation_parameters.xml");

}


float normalize(float * data, size_t size)
{
	float max_elem = 0.0f;
	for (int i = 0; i < size; i++)
	{
		max_elem = std::max(max_elem, data[i]);
	}
	for (int i = 0; i < size; i++)
	{
		data[i] /= max_elem;
	}
	return max_elem;
}

float get_max(float * data, size_t size)
{
	float max_elem = 0.0f;
	for (int i = 0; i < size; i++)
	{
		max_elem = std::max(max_elem, data[i]);
	}
	return max_elem;
}

float average(float * data, size_t size)
{
	float avg = 0.0f;
	for (int i = 0; i < size; i++)
	{
		avg += data[i];
	}
	avg /= size;
	return avg;
}


void FullBSSRDFGenerator::trace()
{
	static int frame = 0;

	if (!mPaused)
	{

		m_context["frame"]->setUint(frame);

		mCurrentBssrdfRenderer->load_data();

		if (mCurrentRenderMode == RenderMode::RENDER_BSSRDF)
		{
			mCurrentBssrdfRenderer->render();
			m_context["show_false_colors"]->setUint(mShowFalseColors);
			m_context["interpolation"]->setUint(mInterpolation);
			m_context["reference_bssrdf_fresnel_mode"]->setInt(mFresnelMode);

			void* source = mCurrentBssrdfRenderer->get_output_buffer()->map();

			memcpy(mCurrentHemisphereData, source, mCurrentBssrdfRenderer->get_size().x*mCurrentBssrdfRenderer->get_size().y * sizeof(float));
			if (!mFastMode)
			{
				void* dest = mBSSRDFBufferTexture->map();
				memcpy(dest, mCurrentHemisphereData, mCurrentBssrdfRenderer->get_size().x*mCurrentBssrdfRenderer->get_size().y * sizeof(float));


				mCurrentAverage = average((float *) dest, (int) mCurrentBssrdfRenderer->get_storage_size());
                mCurrentMax = get_max((float *) dest, (int) mCurrentBssrdfRenderer->get_storage_size());

				mBSSRDFBufferTexture->unmap();
			}
			mCurrentBssrdfRenderer->get_output_buffer()->unmap();
		}

        float m = mScaleMultiplier;
        if(mNormalize)
            m = 1.0f / mCurrentMax;

		m_context["reference_scale_multiplier"]->setFloat(m);


		if (!mFastMode)
		{
			RTsize w, h;
			result_buffer->getSize(w, h);

            float albedo, ext, g, n2_over_n1;
            mCurrentBssrdfRenderer->get_material_parameters(albedo, ext, g, n2_over_n1);

			m_context["ior"]->setFloat(n2_over_n1);
			m_context->launch(entry_point_output, w, h);
		}

        auto time1 = currentTime();
        frame++;
        auto dur = time1 - mCurrentStartTime;
		update_rendering(std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count() / 1e9);
	}
}

optix::Buffer FullBSSRDFGenerator::get_output_buffer()
{
	return result_buffer;
}

void FullBSSRDFGenerator::post_draw_callback()
{
	if (gui == nullptr)
		return;
	gui->start_window("Ray tracing demo", 20, 20, 500, 600);

	ImmediateGUIDraw::InputFloat("Reference scale multiplier", &mScaleMultiplier);

	if (mCurrentRenderMode == RENDER_BSSRDF)
	{
		ImmediateGUIDraw::Text("Current max %e, Current average %e", mCurrentMax, mCurrentAverage);
	}

	ImmediateGUIDraw::Checkbox("Show false colors", (bool*)&mShowFalseColors);
	ImmediateGUIDraw::SameLine();
    ImmediateGUIDraw::Checkbox("Lerp values", (bool*)&mInterpolation);
	ImmediateGUIDraw::Checkbox("Pause",&mPaused);
    ImmediateGUIDraw::SameLine();
    if(ImmediateGUIDraw::Checkbox("Debug",&mDebug))
    {
        m_context->setPrintEnabled(mDebug);
    }

	ImmediateGUIDraw::Checkbox("Fast Mode", &mFastMode);
    ImmediateGUIDraw::SameLine();
    ImmediateGUIDraw::Checkbox("Normalize", &mNormalize);

	ImmediateGUIDraw::Text("Render mode:        ");
	ImmediateGUIDraw::SameLine();
	bool changed_render = ImmediateGUIDraw::RadioButton("Render BSSRDF", (int*)&mCurrentRenderMode, RenderMode::RENDER_BSSRDF);
	ImmediateGUIDraw::SameLine();
	changed_render |= ImmediateGUIDraw::RadioButton("Show Existing", (int*)&mCurrentRenderMode, RenderMode::SHOW_EXISTING_BSSRDF);
	if(changed_render)
		set_render_mode(mCurrentRenderMode, mSimulate);

	if (mCurrentRenderMode == RENDER_BSSRDF)
	{
		static int sim = mSimulate;
		ImmediateGUIDraw::Text("Simulation type:    ");
		ImmediateGUIDraw::SameLine();
		bool changed_simul = ImmediateGUIDraw::RadioButton("Path traced", &sim, 1);
		ImmediateGUIDraw::SameLine();
		changed_simul |= ImmediateGUIDraw::RadioButton("Dipole", &sim, 0);
		if (changed_simul)
			set_render_mode(mCurrentRenderMode, sim == 0 ? false : true);


		static OutputShape::Type shape = mCurrentBssrdfRenderer->get_shape();
        static optix::uint2 size = mCurrentBssrdfRenderer->get_size();

		ImmediateGUIDraw::Text("Storage shape:   ");
		ImmediateGUIDraw::SameLine();
		bool changed_shape = ImmediateGUIDraw::RadioButton("Hemisphere", (int*)&shape, OutputShape::HEMISPHERE);
		ImmediateGUIDraw::SameLine();
		changed_shape |= ImmediateGUIDraw::RadioButton("Plane", (int*)&shape, OutputShape::PLANE);
		if (changed_shape)
		{
			mCurrentBssrdfRenderer->set_shape(shape);
			mCurrentBssrdfRenderer->set_size(mCurrentBssrdfRenderer->get_default_size(shape));
			delete[] mCurrentHemisphereData;
			mCurrentHemisphereData = new float[mCurrentBssrdfRenderer->get_storage_size()];
			mBSSRDFBufferTexture->setSize(mCurrentBssrdfRenderer->get_size().x, mCurrentBssrdfRenderer->get_size().y);
		}

        bool changed_size = ImmediateGUIDraw::InputInt2("Simulation size", (int*)&size, ImGuiInputTextFlags_EnterReturnsTrue);
        if (changed_size)
        {
            mCurrentBssrdfRenderer->set_size(size);
            delete[] mCurrentHemisphereData;
            mCurrentHemisphereData = new float[mCurrentBssrdfRenderer->get_storage_size()];
            mBSSRDFBufferTexture->setSize(mCurrentBssrdfRenderer->get_size().x, mCurrentBssrdfRenderer->get_size().y);
        }


		ImmediateGUIDraw::Text("Fresnel coefficient:");
		ImmediateGUIDraw::SameLine();
		ImmediateGUIDraw::RadioButton("Fresnel + BSSRDF", &mFresnelMode, BSSRDF_RENDER_MODE_FULL_BSSRDF);
		ImmediateGUIDraw::SameLine();
		ImmediateGUIDraw::RadioButton("Fresnel", &mFresnelMode, BSSRDF_RENDER_MODE_FRESNEL_OUT_ONLY);
		ImmediateGUIDraw::SameLine();
		ImmediateGUIDraw::RadioButton("BSSRDF", &mFresnelMode, BSSRDF_RENDER_MODE_REMOVE_FRESNEL);

		if (!mSimulate)
		{
			static ScatteringDipole::Type dipole =  std::dynamic_pointer_cast<BSSRDFRendererModel>(mCurrentBssrdfRenderer)->get_dipole();
			if (BSSRDF::bssrdf_selection_gui(dipole))
			{
				std::dynamic_pointer_cast<BSSRDFRendererModel>(mCurrentBssrdfRenderer)->set_dipole(dipole);
			}
		}

		mCurrentBssrdfRenderer->on_draw(BSSRDFRenderer::SHOW_ALL);

		const char * opts[6] = {"A","B","C","D","E", "F"};
		static int def = 0;
		if (ImmediateGUIDraw::Combo("Default material", &def, opts, 6,6))
		{
			float theta_i; float r; float theta_s; float albedo; float extinction; float g; float n2_over_n1;
			get_default_material(opts[def], theta_i, r, theta_s, albedo, extinction, g, n2_over_n1);
			optix::float2 ts = optix::make_float2(theta_s, theta_s + 7.5f);
			optix::float2 rs = optix::make_float2(r, r + 1.0f);

			mCurrentBssrdfRenderer->set_geometry_parameters(theta_i, rs, ts);
			mCurrentBssrdfRenderer->set_material_parameters(albedo, 1, g, n2_over_n1);
		}
	}
	else if(mCurrentRenderMode == SHOW_EXISTING_BSSRDF)
	{
		if (mLoadedBSSRDF != nullptr)
		{
			static ParameterStateNew index = mLoadedBSSRDF->begin_index();
			for (int i = 0; i < 6; i++)
			{
				std::string s;
				for (int k = 0; k < mLoadedBSSRDF->get_parameters_copy().at(i).size(); k++)
				{
					s += std::to_string(mLoadedBSSRDF->get_parameters_copy().at(i)[k]) + '\0';
				}
                auto ns = EmpiricalBSSRDF::get_parameter_names(mCurrentBssrdfRenderer->get_shape());

                if (ImmediateGUIDraw::Combo(ns[i].c_str(), (int*)index.data()[i], s.c_str(), (int)mLoadedBSSRDF->get_parameters_copy().at(i).size()))
				{
					float * data = (float*)mExternalBSSRDFBuffer->map();
                    mLoadedBSSRDF->load_hemisphere(data, index);
					mExternalBSSRDFBuffer->unmap();

					float theta_i; optix::float2 r; optix::float2 theta_s; float albedo;  float g; float eta;
                    mLoadedBSSRDF->get_parameters(index, theta_i, r, theta_s, albedo, g, eta);
					// Second one is ignored anyways in this configuration...
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
			if (Dialogs::open_file_dialog(filePath))
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
			for (int i = 0; i < mSimulationBSSRDF->get_parameters_copy().size(); i++)
			{
				data.push_back(new char[256]);
				std::string s = gui_string(mSimulationBSSRDF->get_parameters_copy()[i]);
				s.copy(&data[i][0], s.size());
				data[i][s.size()] = '\0';
			}
		}

		for (int i = 0; i < 6; i++)
		{
            auto ns = EmpiricalBSSRDF::get_parameter_names(mCurrentBssrdfRenderer->get_shape());
			if (ImGui::InputText((std::string("Simulated ") + ns[i]).c_str(), data[i], 256, ImGuiInputTextFlags_EnterReturnsTrue))
			{
				std::vector<float> c = tovalue<std::vector<float>>(std::string(data[i]));
				std::string s = gui_string(c);
				s.copy(&data[i][0], s.size());
                mSimulationBSSRDF->set_parameter_values(i, c);
			}
		}

		if (ImmediateGUIDraw::Button("Reset parameters"))
		{
            mSimulationBSSRDF->reset_parameters();
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
			const float progress_total = mSimulationState.flatten() / (float)mSimulationBSSRDF->get_material_geometry_combinations();
			const auto dims = mSimulationBSSRDF->get_material_geometry_dimensions();
			const size_t material_size = dims[theta_i_index] * dims[theta_s_index] * dims[r_index];
			const float progress_bssrdf = (mSimulationState.flatten() % material_size) / (float)material_size;
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
}

void FullBSSRDFGenerator::start_rendering()
{
    auto parameters = mSimulationBSSRDF->get_parameters_copy();
	Logger::info << "Simulation started. " << std::endl;
	Logger::info << current_render_task->to_string() << std::endl;
	Logger::info << "Destination file: " << current_render_task->get_destination_file() << std::endl;
	Logger::info << "Eta:     " << tostring(parameters[eta_index]) << std::endl;
	Logger::info << "G:       " << tostring(parameters[g_index]) << std::endl;
	Logger::info << "Albedo:  " << tostring(parameters[albedo_index]) << std::endl;
	Logger::info << "Theta s: " << tostring(parameters[theta_s_index]) << std::endl;
	Logger::info << "R:       " << tostring(parameters[r_index]) << std::endl;
	Logger::info << "Theta i: " << tostring(parameters[theta_i_index]) << std::endl;
	current_render_task->start();

    mSimulationBSSRDF->set_shape(mCurrentBssrdfRenderer->get_shape());
    mSimulationBSSRDF->set_filename(current_render_task->get_destination_file());
    mSimulationBSSRDF->set_hemisphere_size(mCurrentBssrdfRenderer->get_size().x, mCurrentBssrdfRenderer->get_size().x);

	mSimulationState = mSimulationBSSRDF->begin_index();

	if (mSimulate)
	{
		std::dynamic_pointer_cast<BSSRDFRendererSimulated>(mCurrentBssrdfRenderer)->set_samples(mSimulationSamplesPerFrame);
		std::dynamic_pointer_cast<BSSRDFRendererSimulated>(mCurrentBssrdfRenderer)->set_max_iterations(mSimulationMaxIterations);
	}

    mCurrentBssrdfRenderer->set_read_only(true);
	float extinction = 1.0f;
	float theta_i; optix::float2 r; optix::float2 theta_s; float albedo;  float g; float eta;
    mSimulationBSSRDF->get_parameters(mSimulationState, theta_i, r, theta_s, albedo, g, eta);

	mCurrentBssrdfRenderer->set_geometry_parameters(theta_i, r, theta_s);
	mCurrentBssrdfRenderer->set_material_parameters(albedo, extinction, g, eta);
	mFastMode = false;
	mPaused = false;
    mSimulationBSSRDF->save_header(mSimulationBSSRDF->get_filename());
    mCurrentStartTime = currentTime();

}

void FullBSSRDFGenerator::update_rendering(float time_past)
{
	if (current_render_task->is_active())
	{
		if (current_render_task->is_finished())
		{
			Logger::info << "Simulation frame " << mSimulationState.flatten() + 1 << "/" << mSimulationBSSRDF->get_material_geometry_combinations() << " complete. ";
			if (mSimulate)
			{
				Logger::info << std::dynamic_pointer_cast<BSSRDFRendererSimulated>(mCurrentBssrdfRenderer)->get_samples() << " samples. ";
			}
			float theta_i; optix::float2 r; optix::float2 theta_s; float albedo;  float g; float eta;
            mSimulationBSSRDF->get_parameters(mSimulationState, theta_i, r, theta_s, albedo, g, eta);
			Logger::info <<  std::endl;

            Logger::info << "Parameters: r " << r.x << " " << r.y << "; " << "theta_s " << theta_s.x << " " << theta_s.y << std::endl;
			Logger::info << "Index: " << mSimulationState.tostring() << "(simulation) Parameters: eta " << eta << " g " << g << " albedo " << albedo << std::endl;
			std::cout << "Average: " << std::scientific << average(mCurrentHemisphereData, mSimulationBSSRDF->get_hemisphere_size()) << std::defaultfloat << std::endl;
            mSimulationBSSRDF->set_hemisphere(mCurrentHemisphereData, mSimulationState);
            mSimulationState = mSimulationState.next(mSimulationState);
			float extinction = 1.0f;

			if (!mSimulationState.is_valid())
			{
				end_rendering();
                mCurrentStartTime = currentTime();
			}
			else
			{
                mSimulationBSSRDF->get_parameters(mSimulationState, theta_i, r, theta_s, albedo, g, eta);
				mCurrentBssrdfRenderer->set_geometry_parameters(theta_i, r, theta_s);
				mCurrentBssrdfRenderer->set_material_parameters(albedo, extinction, g, eta);
				current_render_task->start();
                mCurrentStartTime = currentTime();
			}

		}
		else
		{
			current_render_task->update_absolute(time_past);
		}
	}
}

void FullBSSRDFGenerator::end_rendering()
{
	current_render_task->end();
	mCurrentBssrdfRenderer->set_read_only(false);
    mPaused = true;
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
	mLoadedBSSRDF = std::make_unique<EmpiricalBSSRDF>();
    mLoadedBSSRDF->load_header(file);
	auto dim1 = mLoadedBSSRDF->get_dimension_1();
	auto dim2 = mLoadedBSSRDF->get_dimension_2();

	auto sl = mLoadedBSSRDF->get_material_slice_size();
	float * d2 = new float[sl];
    mLoadedBSSRDF->load_material_slice(d2, {0,0,0});
	float mx = get_max(d2, sl);
	Logger::info << "Calculated max: " << mx << std::endl;


	mExternalBSSRDFBuffer->setSize(dim1,dim2);
	float * data = (float*)mExternalBSSRDFBuffer->map();
    mLoadedBSSRDF->load_hemisphere(data, mLoadedBSSRDF->begin_index());
	mExternalBSSRDFBuffer->unmap();
    set_render_mode(mCurrentRenderMode, mSimulate);
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

    OutputShape::Type shape = OutputShape::HEMISPHERE;
	if(m == RENDER_BSSRDF)
    {
        mCurrentBssrdfRenderer = mSimulate ? std::static_pointer_cast<BSSRDFRenderer>(mBssrdfReferenceSimulator)
                                           : std::static_pointer_cast<BSSRDFRenderer>(mBssrdfModelSimulator);
        shape = mCurrentBssrdfRenderer->get_shape();
    }
    else
    {
        if(mLoadedBSSRDF != nullptr)
            shape = mLoadedBSSRDF->get_shape();
    }
    m_context["reference_bssrdf_output_shape"]->setUserData(sizeof(OutputShape::Type), &shape);
}


void FullBSSRDFGenerator::set_render_task(std::unique_ptr<RenderTask>& task)
{
	if (!current_render_task->is_active())
		current_render_task = std::move(task);
	else
		Logger::error << "Wait of end of current task before setting a new one." << std::endl;
}

Camera *FullBSSRDFGenerator::get_camera()
{
	CameraParameters c;
	return new Camera(m_context, c);
}

void FullBSSRDFGenerator::load_parameters(const std::string &config_file)
{
    if (!exists(config_file.c_str()))
        return;
    cereal::XMLInputArchiveOptix archive(m_context, config_file);
    archive(cereal::make_nvp("data_folder", Folders::data_folder));
    archive(cereal::make_nvp("ptx_folder", Folders::ptx_path));
	std::string tmp;
    archive(cereal::make_nvp("eta_values", tmp));
    mSimulationBSSRDF->set_parameter_values(eta_index, tovalue<std::vector<float>>(tmp));
    archive(cereal::make_nvp("albedo_values",tmp));
    mSimulationBSSRDF->set_parameter_values(albedo_index, tovalue<std::vector<float>>(tmp));
    mSimulationBSSRDF->set_parameter_values(g_index, tovalue<std::vector<float>>(tmp));
    archive(cereal::make_nvp("g_values", tmp));
}

void FullBSSRDFGenerator::save_parameters(const std::string &config_file)
{
    cereal::XMLOutputArchiveOptix archive(config_file);
    archive(cereal::make_nvp("data_folder", Folders::data_folder));
    archive(cereal::make_nvp("ptx_folder", Folders::ptx_path));
    archive(cereal::make_nvp("eta_values", tostring(mSimulationBSSRDF->get_parameters_copy()[eta_index])));
    archive(cereal::make_nvp("albedo_values", tostring(mSimulationBSSRDF->get_parameters_copy()[albedo_index])));
    archive(cereal::make_nvp("g_values", tostring(mSimulationBSSRDF->get_parameters_copy()[g_index])));
}
