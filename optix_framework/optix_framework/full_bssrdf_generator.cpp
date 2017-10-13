#include "full_bssrdf_generator.h"
#include "optix_utils.h"
#include "immediate_gui.h"
#include <bssrdf_creator.h>
#include "parserstringhelpers.h"
#include <fstream>
#include "parameter_parser.h"
#include "folders.h"
#include "dialogs.h"


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

	creator = std::make_unique<ReferenceBSSRDF>(m_context, optix::make_uint2(160, 40), (int)10e5);
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

}


void normalize(float * data, int size)
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
	m_context["frame"]->setInt(frame);


	creator->load_data();

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

	RTsize w, h;
	result_buffer->getSize(w, h);
	m_context->launch(entry_point_output, w, h);
	
	frame++;
	update_rendering();
}

optix::Buffer FullBSSRDFGenerator::get_output_buffer()
{
	return result_buffer;
}


void FullBSSRDFGenerator::post_draw_callback()
{
	gui->start_draw();
	gui->start_window("Ray tracing demo", 20, 20, 500, 600);
	
	ImmediateGUIDraw::InputFloat("Reference scale multiplier", &mScaleMultiplier);

	ImmediateGUIDraw::Checkbox("Show false colors", (bool*)&mShowFalseColors);

	creator->on_draw(true);

	static std::string theta_i_s = tostring(mParameters.theta_i_v);
	if (ImGui::InputText("Theta i", &theta_i_s[0], theta_i_s.size()))
	{
		std::vector<float> c = tovalue<std::vector<float>>(theta_i_s);
		mParameters.theta_i_v.clear();
		mParameters.theta_i_v.insert(mParameters.theta_i_v.begin(), c.begin(), c.end());
	}

	if (ImmediateGUIDraw::Button("Start Simulation"))
	{
		start_rendering();
	}

	gui->end_window();
	gui->end_draw();
}

void FullBSSRDFGenerator::start_rendering()
{
	is_rendering = true;

	size_t total_size = mParameters.size() * sizeof(float) * creator->get_storage_size();

	
	std::ofstream of;
	of.open(mFilePath);
	of.seekp(total_size - 1);
	of.put('\0');
	of.close();

	mState = ParameterState(0,0,0,0,0,0);
	creator->set_samples((int)1e8);
}



void FullBSSRDFGenerator::update_rendering()
{
	if (is_rendering)
	{
		Logger::info << mState.tostring() << std::endl;
		float extinction = 1.0f;
		float theta_i; float r; float theta_s; float albedo;  float g; float eta;
		mParameters.get_parameters(mState, theta_i, r, theta_s, albedo, g, eta);

		creator->set_geometry_parameters(theta_i, r, theta_s);
		creator->set_material_parameters(albedo, extinction, g, eta);

		
		write_hemisphere(mFilePath, mParameters.flatten(mState), mCurrentHemisphereData, creator->get_storage_size() * sizeof(float));

		mState = mParameters.next(mState);
		if (!mParameters.is_valid(mState))
		{
			is_rendering = false;
		}
	}
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

void FullBSSRDFGenerator::write_hemisphere(std::string file, size_t flattened_index, float * hemisphere, size_t hemisphere_size_bytes)
{
	size_t pos = flattened_index * hemisphere_size_bytes;
	std::ofstream of;
	of.open(file, std::ofstream::out | std::ofstream::in | std::ofstream::binary);
	of.seekp(pos);
	of.write(reinterpret_cast<char*>(hemisphere), hemisphere_size_bytes);
	of.close();
}

void FullBSSRDFGenerator::clean_up()
{
	delete[] mCurrentHemisphereData;
}


