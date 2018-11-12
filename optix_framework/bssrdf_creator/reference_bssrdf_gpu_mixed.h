#pragma once
#include "reference_bssrdf.h"
#include "bssrdf_creator.h"
#include "reference_bssrdf_gpu.h"

#define IMPROVED_ENUM_NAME ReferenceRendererPreset
#define IMPROVED_ENUM_LIST ENUMITEM_VALUE(MCML_REFERENCE,0) ENUMITEM_VALUE(DONNER_ET_AL,1)  ENUMITEM_VALUE(CONNECTIONS_CORRECTED, 2) ENUMITEM_VALUE(CONNECTIONS_BIASED,3) ENUMITEM_VALUE(MIXED_BIAS_REDUCTION, 4)
#include "improved_enum.inc"


class ReferenceBSSRDFGPUMixed : public BSSRDFRendererSimulated
{
public:
    ReferenceBSSRDFGPUMixed(optix::Context & ctx, const OutputShape::Type shape = DEFAULT_SHAPE, const optix::int2 & shape_size = optix::make_int2(-1), const unsigned int samples = (int)1e8) : BSSRDFRendererSimulated(ctx, shape, shape_size, samples)
    {
        mRenderer1 = std::make_unique<ReferenceBSSRDFGPU>(ctx, shape, shape_size, samples);
        mRenderer2 = std::make_unique<ReferenceBSSRDFGPU>(ctx, shape, shape_size, samples);
    }

    enum ReferenceRendererToUse
    {
        RENDERER_1 = 0,
        RENDERER_2 = 1,
        RENDERER_BOTH = 2
    };


    void init() override;
    void load_data() override;
    void render() override;
    bool on_draw(unsigned int flags) override;
    void reset() override;
    void set_shape(OutputShape::Type shape) override;
    void set_size(optix::uint2 shape) override;

    void set_geometry_parameters(float theta_i, optix::float2 r, optix::float2 theta_s) override;
    void set_material_parameters(float albedo, float extinction, float g, float eta) override;
    void set_preset(ReferenceRendererPreset::Type preset);


    optix::Buffer get_output_buffer() override;

    std::unique_ptr<ReferenceBSSRDFGPU> mRenderer1 = nullptr;
    std::unique_ptr<ReferenceBSSRDFGPU> mRenderer2 = nullptr;
    ReferenceRendererToUse mRendererToUse = RENDERER_1;
    ReferenceRendererToUse mRendererToShow = RENDERER_1;

    int entry_point_sum = -1;
    optix::Buffer mBSSRDFSumBuffer;
    ReferenceRendererPreset::Type mPreset = ReferenceRendererPreset::MCML_REFERENCE;
};

