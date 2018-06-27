#include "empirical_bssrdf_host.h"
#include "logger.h"
#include "bssrdf_loader.h"
#include "optix_host_utils.h"
#include "scattering_properties.h"
#include "empirical_bssrdf_common.h"
#include "optix_math.h"
#include "folders.h"

void EmpiricalBSSRDF::prepare_buffers()
{
    mBSSRDFLoader = std::make_unique<BSSRDFImporter>(mBSSRDFFile);

    // First, allocating data buffers, one per wavelength.
    size_t s = mBSSRDFLoader->get_material_slice_size();
    for(int i = 0; i < 3; i++)
    {
        auto buf = create_buffer<float>(mContext, RT_BUFFER_INPUT_OUTPUT, s);
        fill_buffer<float>(buf, 0.0f);
        mDataBuffers.buffers[i] = buf->getId();
    }

    // Parameter buffers. Some extra care is needed for theta_o, phi_o, since in the reader they are dealt with differently.
    std::vector<size_t> full_dimensions;
    mBSSRDFLoader->get_dimensions(full_dimensions);
    mParameterSizeBuffer = create_buffer<int>(mContext, RT_BUFFER_INPUT, 5);
    int* buf = reinterpret_cast<int*>(mParameterSizeBuffer->map());
    buf[0] = (int)full_dimensions[theta_s_index];
    buf[1] = (int)full_dimensions[r_index];
    buf[2] = (int)full_dimensions[theta_i_index];
    buf[4] = (int)full_dimensions[dim_1_index];
    buf[3] = (int)full_dimensions[dim_2_index];
    mParameterSizeBuffer->unmap();

    auto parameters = mBSSRDFLoader->get_parameters();
    mParameterBuffers.buffers[0] = create_and_initialize_buffer(mContext, parameters[theta_s_index] )->getId();
    mParameterBuffers.buffers[1] = create_and_initialize_buffer(mContext,parameters[r_index] )->getId();
    mParameterBuffers.buffers[2] = create_and_initialize_buffer(mContext, parameters[theta_i_index] )->getId();
    mParameterBuffers.buffers[4] = create_and_initialize_buffer(mContext, parameters[dim_1_index] )->getId();
    mParameterBuffers.buffers[3] = create_and_initialize_buffer(mContext, parameters[dim_2_index] )->getId();
    mManager = std::make_unique<BSSRDFParameterManager>(parameters);
    mInitialized = true;
}

EmpiricalBSSRDF::EmpiricalBSSRDF(optix::Context & context): BSSRDF(context, ScatteringDipole::EMPIRICAL_BSSRDF)
{
    mDataBuffers.buffers[0] = 0;
    mDataBuffers.buffers[1] = 0;
    mDataBuffers.buffers[2] = 0;

    mBSSRDFFile = Folders::data_folder + "/bssrdf/test.bssrdf";
}


void EmpiricalBSSRDF::load(const optix::float3 &relative_ior, const ScatteringMaterialProperties &props)
{
    if(!mInitialized)
    {
        float ior = dot(relative_ior, optix::make_float3(1)) / 3.0f;

        Logger::info << "Ab    : " << props.absorption.x << " " << props.absorption.y << " "<< props.absorption.z << std::endl;
        Logger::info << "Scatt : " << props.scattering.x << " " << props.scattering.y << " "<< props.scattering.z << std::endl;
        Logger::info << "Albedo: " << props.albedo.x << " " << props.albedo.y << " "<< props.albedo.z << std::endl;
        Logger::info << "Ext   : " << props.extinction.x << " " << props.extinction.y << " "<< props.extinction.z << std::endl;
        Logger::info << "G   : " << props.meancosine.x << " " << props.meancosine.y << " "<< props.meancosine.z << std::endl;
        Logger::info << "Eta   : " << ior << std::endl;
        Logger::info << "Loading new material for empirical bssrdf..." << std::endl;
        prepare_buffers();

        mContext["empirical_bssrdf_parameters"]->setUserData(sizeof(EmpiricalParameterBuffer), &mParameterBuffers);
        auto id1 = rtBufferId<int>(mParameterSizeBuffer->getId());
        mContext["empirical_bssrdf_parameters_size"]->setUserData(sizeof(rtBufferId<int>), &id1);

        std::vector<size_t> state;
        bool found = true;
        float rel_ior = dot(relative_ior, optix::make_float3(1)) / 3.0f;

        for(int i = 0; i < 3; i++)
        {
            // We find the corresponding index for the material properties.
            mManager->get_material_index(optix::getByIndex(props.albedo,i), optix::getByIndex(props.meancosine,i), rel_ior, state);
            optix::Buffer buf = mContext->getBufferFromId(mDataBuffers.buffers[i].getId());
            RTsize w;
            buf->getSize(w);

            float * data = reinterpret_cast<float*>(buf->map());
            found &= mBSSRDFLoader->load_material_slice(data, state);

            float mx = 0.0f;
            for(int i = 0; i < w; i++)
            {
                mx = std::max(mx, data[i]);
                if(std::isnan(data[i]))
                {
                    Logger::error << "Nan Found!" << std::endl;
                }
            }
            Logger::info << "Loaded bssrdf max " << mx << std::endl;

            buf->unmap();
            mContext["empirical_buffer_size"]->setInt(w);
        }

        mContext["empirical_buffer"]->setUserData(sizeof(EmpiricalDataBuffer), &mDataBuffers);
        Logger::info << "Loading complete." << std::endl;
        if(!found)
        {
            mInitialized = false;
        }
        mContext["empirical_bssrdf_correction"]->setFloat(mCorrection);
        mContext["non_planar_geometry_handle"]->setUserData(sizeof(EmpiricalBSSRDFNonPlanarity::Type), &mNonPlanarSurfacesHandles);

    }
}

bool EmpiricalBSSRDF::on_draw()
{
    bool changed = false;
    if(ImmediateGUIDraw::InputFloat("Correction" ,&mCorrection))
    {
        mContext["empirical_bssrdf_correction"]->setFloat(mCorrection);
        changed = true;
    }
    if(ImmediateGUIDraw::Combo("Interpolation", (int*)&mInterpolation, "Nearest\0Linear", 2))
    {
        mContext["empirical_bssrdf_interpolation"]->setUint(mInterpolation);
        changed = true;
    }

    std::string data = "";
    auto f = EmpiricalBSSRDFNonPlanarity::first();
    while(f != EmpiricalBSSRDFNonPlanarity::NotValidEnumItem)
    {
        data += EmpiricalBSSRDFNonPlanarity::to_string(f) + '\0';
        f = EmpiricalBSSRDFNonPlanarity::next(f);
    }

    if(ImmediateGUIDraw::Combo("Normal handle", (int*)&mNonPlanarSurfacesHandles, &data[0], EmpiricalBSSRDFNonPlanarity::count()))
    {
        mContext["non_planar_geometry_handle"]->setUserData(sizeof(EmpiricalBSSRDFNonPlanarity::Type), &mNonPlanarSurfacesHandles);
        changed = true;
    }
    return changed;
}
