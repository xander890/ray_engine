#include "presampled_surface_bssrdf.h"
#include "material_library.h"

void PresampledSurfaceBssrdf::initialize_shader(optix::Context ctx, const ShaderInfo& shader_info)
{
    Shader::initialize_shader(ctx, shader_info);
    //in static constructor

    std::string ptx_path = get_path_ptx("sample_camera.cu");
    Program ray_gen_program = context->createProgramFromPTXFile( ptx_path, "sample_camera" );
    entry_point = context->getEntryPointCount();
    context->setEntryPointCount(entry_point + 1);
    context->setRayGenerationProgram(entry_point , ray_gen_program);
    
    // We tell optix we "are doing stuff" in the context with these names --> otherwise compilation fails
    Buffer empty_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 1);
    Buffer empty_bufferi3 = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, 1);
    
    context["sampling_vertex_buffer"]->setBuffer(empty_buffer);
    context["sampling_normal_buffer"]->setBuffer(empty_buffer);
    context["sampling_vindex_buffer"]->setBuffer(empty_bufferi3);
    context["sampling_nindex_buffer"]->setBuffer(empty_bufferi3);

    m_samples = context->createBuffer(RT_BUFFER_OUTPUT);
    m_samples->setFormat(RT_FORMAT_USER);
    m_samples->setElementSize(sizeof(PositionSample));
    m_samples->setSize(SAMPLES_FRAME);
    context["sampling_output_buffer"]->setBuffer(m_samples);
}

void PresampledSurfaceBssrdf::load_into_mesh(Mesh& object)
{
    Geometry g = object.mGeometry;    
    // precompute triangle area cdf
    Buffer v_buff = g["vertex_buffer"]->getBuffer();
    Buffer v_idx_buff = g["vindex_buffer"]->getBuffer();
    float3* vbuffer_data = static_cast<float3*>(v_buff->map());
    int3* vbuffer_idx_data = static_cast<int3*>(v_idx_buff->map());
    RTsize buffer_width;
    v_idx_buff->getSize(buffer_width);
    unsigned int triangles = static_cast<unsigned int>(buffer_width);

    float* cdf = new float[triangles];
    float totalArea = 0.0f;
    for (unsigned int i = 0; i < triangles; ++i)
    {
    	int3 index = vbuffer_idx_data[i];
    	float3 v0 = vbuffer_data[index.x];
    	float3 v1 = vbuffer_data[index.y];
    	float3 v2 = vbuffer_data[index.z];
    	float area = 0.5f*length(cross(v1 - v0, v2 - v0));
    	cdf[i] = area + (i > 0 ? cdf[i - 1] : 0.0f);
    	totalArea += area;
    }
    v_idx_buff->unmap();
    v_buff->unmap();
    object.mGeometryInstance["total_area"]->setFloat(totalArea);
    std::cout << "Area: " << totalArea << std::endl;

    for (unsigned int i = 0; i < triangles; ++i)
    {
    	cdf[i] /= totalArea;
    }
    Buffer cdf_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, triangles);
    memcpy(cdf_buffer->map(), cdf, triangles * sizeof(float));
        object.mGeometryInstance["area_cdf"]->setBuffer(cdf_buffer);
    cdf_buffer->unmap();
    delete[] cdf;

	Shader::load_into_mesh(object);
}


void PresampledSurfaceBssrdf::pre_trace_mesh(Mesh& obj)
{
    Geometry g = obj.mGeometry;
    GeometryInstance object = obj.mGeometryInstance;
    context["sampling_vertex_buffer"]->setBuffer(g["vertex_buffer"]->getBuffer());
    context["sampling_normal_buffer"]->setBuffer(g["normal_buffer"]->getBuffer());
    context["sampling_vindex_buffer"]->setBuffer(g["vindex_buffer"]->getBuffer());
    context["sampling_nindex_buffer"]->setBuffer(g["nindex_buffer"]->getBuffer());
    context["bssrdf_enabled"]->setUint(0);	// Disabling BSSRDF computation on sample collection.
    context->launch(entry_point, SAMPLES_FRAME);
    context["bssrdf_enabled"]->setUint(1);
}
