#include <optix_host_utils.h>
#include <parsing_utils.h>
#include "reference_bssrdf_gpu_mixed.h"
#include "immediate_gui.h"

void ReferenceBSSRDFGPUMixed::init()
{
    mRenderer2->init();
    mRenderer1->init();

    std::string ptx_path = Folders::get_path_to_ptx("reference_bssrdf_gpu_mixed.cu");
    optix::Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, "post");

    if (entry_point_sum == -1)
    {
        entry_point_sum = add_entry_point(context, ray_gen_program);
    }

    if (mBSSRDFSumBuffer.get() == nullptr)
    {
        mBSSRDFSumBuffer = create_buffer<float>(context, RT_BUFFER_INPUT_OUTPUT, mShapeSize.x * mShapeSize.y);
        mBSSRDFSumBuffer->setFormat(RT_FORMAT_FLOAT);
        mBSSRDFSumBuffer->setSize(mShapeSize.x, mShapeSize.y);
    }

    BufPtr2D<float> ptr1 = BufPtr2D<float>(mRenderer1->get_output_buffer()->getId());
    context["reference_resulting_flux_1"]->setUserData(sizeof(BufPtr2D<float>), &ptr1);
    BufPtr2D<float> ptr2 = BufPtr2D<float>(mRenderer2->get_output_buffer()->getId());
    context["reference_resulting_flux_2"]->setUserData(sizeof(BufPtr2D<float>), &ptr2);
    BufPtr2D<float> ptr_sum = BufPtr2D<float>(mBSSRDFSumBuffer->getId());
    context["reference_resulting_flux_total"]->setUserData(sizeof(BufPtr2D<float>), &ptr_sum);
    mRenderer2->set_read_only(true);
}

void ReferenceBSSRDFGPUMixed::render()
{
    if (mRendererToUse == RENDERER_1 || mRendererToUse == RENDERER_BOTH)
    {
        mRenderer1->load_data();
        mRenderer1->render();
    }
    if (mRendererToUse == RENDERER_2 || mRendererToUse == RENDERER_BOTH)
    {
        mRenderer2->load_data();
        mRenderer2->render();
    }
    if (mRendererToUse == RENDERER_BOTH)
    {
        context->launch(entry_point_sum, mShapeSize.x, mShapeSize.y);
    }
}

bool ReferenceBSSRDFGPUMixed::on_draw(unsigned int flags)
{
    bool changed = false;

    std::string s2 = ReferenceRendererPreset::get_full_string();
    std::vector<std::string> tokens;
    split(tokens, s2, ' ');
    std::vector<const char *> c;
    for (std::string &s : tokens)
    {
        c.push_back(s.c_str());
    }

    static ReferenceRendererPreset::Type rendering_preset;
    if (ImmediateGUIDraw::Combo(GUI_LABEL("Rendering preset", mId), (int *) &rendering_preset, &c[0], c.size(), c.size()))
    {
        changed = true;
        set_preset(rendering_preset);
    }

    changed |= ImmediateGUIDraw::Combo(GUI_LABEL("Renderer to use", mId), (int *) &mRendererToUse, "Renderer 1\0Renderer 2\0Both", 3);
    changed |= ImmediateGUIDraw::Combo(GUI_LABEL("Renderer to show", mId), (int *) &mRendererToShow, "Renderer 1\0Renderer 2\0Both", 3);
    ImmediateGUIDraw::Separator();
    ImmediateGUIDraw::Text("%s", "Renderer 1 options");
    changed |= mRenderer1->on_draw(SHOW_ALL);
    ImmediateGUIDraw::Separator();
    ImmediateGUIDraw::Text("%s", "Renderer 2 options");
    changed |= mRenderer2->on_draw(SHOW_EXTRA_OPTIONS);

    if (changed)
    {
        reset();
    }

    return changed;
}

void ReferenceBSSRDFGPUMixed::load_data()
{

}

void ReferenceBSSRDFGPUMixed::reset()
{
    mRenderer1->reset();
    mRenderer2->reset();

    float theta_i;
    optix::float2 r;
    optix::float2 theta_s;
    float albedo, extinction, g, eta;
    mRenderer1->get_geometry_parameters(theta_i, r, theta_s);
    mRenderer2->set_geometry_parameters(theta_i, r, theta_s);
    mRenderer1->get_material_parameters(albedo, extinction, g, eta);
    mRenderer2->set_material_parameters(albedo, extinction, g, eta);
    mRenderer2->set_bias(mRenderer1->get_bias());
}

void ReferenceBSSRDFGPUMixed::set_geometry_parameters(float theta_i, optix::float2 r, optix::float2 theta_s)
{
    BSSRDFRenderer::set_geometry_parameters(theta_i, r, theta_s);
    mRenderer2->set_geometry_parameters(theta_i, r, theta_s);
    mRenderer1->set_geometry_parameters(theta_i, r, theta_s);
}

void ReferenceBSSRDFGPUMixed::set_material_parameters(float albedo, float extinction, float g, float eta)
{
    BSSRDFRenderer::set_material_parameters(albedo, extinction, g, eta);
    mRenderer2->set_material_parameters(albedo, extinction, g, eta);
    mRenderer1->set_material_parameters(albedo, extinction, g, eta);
}

optix::Buffer ReferenceBSSRDFGPUMixed::get_output_buffer()
{
    switch (mRendererToShow)
    {
        case RENDERER_1: return mRenderer1->get_output_buffer();
        case RENDERER_2: return mRenderer2->get_output_buffer();
        case RENDERER_BOTH: return mBSSRDFSumBuffer;
    }
    return nullptr;
}

void ReferenceBSSRDFGPUMixed::set_preset(ReferenceRendererPreset::Type preset)
{
    switch (preset)
    {

        case ReferenceRendererPreset::MCML_REFERENCE:
        {
            mRendererToUse = RENDERER_1;
            mRendererToShow = RENDERER_1;
            mRenderer1->set_bias_visualization_method(BiasMode::RENDER_ALL);
            mRenderer1->set_cosine_weighted(false);
            mRenderer1->set_integration_method(IntegrationMethod::MCML);
        }
            break;
        case ReferenceRendererPreset::DONNER_ET_AL:
        {
            mRendererToUse = RENDERER_1;
            mRendererToShow = RENDERER_1;
            mRenderer1->set_bias_visualization_method(BiasMode::RENDER_ALL);
            mRenderer1->set_cosine_weighted(false);
            mRenderer1->set_integration_method(IntegrationMethod::CONNECTIONS);
        }
            break;
        case ReferenceRendererPreset::CONNECTIONS_CORRECTED:
        {
            mRendererToUse = RENDERER_1;
            mRendererToShow = RENDERER_1;
            mRenderer1->set_bias_visualization_method(BiasMode::RENDER_ALL);
            mRenderer1->set_cosine_weighted(false);
            mRenderer1->set_integration_method(IntegrationMethod::CONNECTIONS_WITH_FIX);
        }
            break;
        case ReferenceRendererPreset::CONNECTIONS_BIASED:
        {
            mRendererToUse = RENDERER_1;
            mRendererToShow = RENDERER_1;
            mRenderer1->set_bias_visualization_method(BiasMode::BIASED_RESULT);
            mRenderer1->set_cosine_weighted(false);
            mRenderer1->set_integration_method(IntegrationMethod::CONNECTIONS_WITH_FIX);
        }
            break;
        case ReferenceRendererPreset::MIXED_BIAS_REDUCTION:
        {
            mRendererToUse = RENDERER_BOTH;
            mRendererToShow = RENDERER_BOTH;
            mRenderer1->set_bias_visualization_method(BiasMode::BIAS_ONLY);
            mRenderer1->set_cosine_weighted(false);
            mRenderer1->set_integration_method(IntegrationMethod::MCML);
            mRenderer2->set_bias_visualization_method(BiasMode::BIASED_RESULT);
            mRenderer2->set_cosine_weighted(false);
            mRenderer2->set_integration_method(IntegrationMethod::CONNECTIONS_WITH_FIX);
        }
            break;
        default:
        case ReferenceRendererPreset::NotValidEnumItem:break;
    }
}

void ReferenceBSSRDFGPUMixed::set_shape(OutputShape::Type shape)
{
    //BSSRDFRenderer::set_shape(shape);
    mOutputShape = shape;
    mRenderer1->set_shape(shape);
    mRenderer2->set_shape(shape);
}

void ReferenceBSSRDFGPUMixed::set_size(optix::uint2 size)
{
    //BSSRDFRenderer::set_size(size);
    mShapeSize = size;
    mBSSRDFSumBuffer->setSize(mShapeSize.x, mShapeSize.y);
    mRenderer1->set_size(size);
    mRenderer2->set_size(size);
}
