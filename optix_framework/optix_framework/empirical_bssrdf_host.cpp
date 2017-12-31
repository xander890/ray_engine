#include "empirical_bssrdf_host.h"
#include "logger.h"
#include "bssrdf_loader.h"
#include "optix_utils.h"
#include "scattering_properties.h"
#include "empirical_bssrdf_utils.h"

std::vector<float> convert_to_rad(const std::vector<float> & vec)
{
    std::vector<float> n;
    for(auto & v : vec)
    {
        n.push_back(deg2rad(v));
    }
    return n;
}

void EmpiricalBSSRDF::prepare_buffers()
{
    mBSSRDFLoader = std::make_unique<BSSRDFLoader>(mBSSRDFFile);

    // First, allocating data buffers, one per wavelength.
    size_t s = mBSSRDFLoader->get_material_slice_size();
    for(int i = 0; i < 3; i++)
    {
        auto buf = create_buffer<float>(mContext, RT_BUFFER_INPUT_OUTPUT, s);
        fill_buffer<float>(buf, 1.0f);
        mDataBuffers.buffers[i] = buf->getId();
    }

    // Parameter buffers. Some extra care is needed for theta_o, phi_o, since in the reader they are dealt with differently.
    std::vector<size_t> full_dimensions;
    mBSSRDFLoader->get_dimensions(full_dimensions);
    mParameterSizeBuffer = create_buffer<int>(mContext, RT_BUFFER_INPUT, 5);
    size_t phi_o_size = mBSSRDFLoader->get_hemisphere_phi_o();
    size_t theta_o_size = mBSSRDFLoader->get_hemisphere_theta_o();
    int* buf = reinterpret_cast<int*>(mParameterSizeBuffer->map());
    buf[0] = full_dimensions[theta_s_index];
    buf[1] = full_dimensions[r_index];
    buf[2] = full_dimensions[theta_i_index];
    buf[3] = theta_o_size;
    buf[4] = phi_o_size;
    mParameterSizeBuffer->unmap();

    auto parameters = mBSSRDFLoader->get_parameters();
    auto converted_theta_i = convert_to_rad(parameters[theta_i_index]);
    auto converted_theta_s = convert_to_rad(parameters[theta_s_index]);

    mParameterBuffers.buffers[0] = create_and_initialize_buffer(mContext,converted_theta_s )->getId();
    mParameterBuffers.buffers[1] = create_and_initialize_buffer(mContext,parameters[r_index] )->getId();
    mParameterBuffers.buffers[2] = create_and_initialize_buffer(mContext,converted_theta_i )->getId();

    optix::Buffer phioBuffer = create_buffer<float>(mContext, RT_BUFFER_INPUT, phi_o_size);
    float * pbuf = reinterpret_cast<float*>(phioBuffer->map());
    for(int i = 0; i < phi_o_size; i++)
    {
        pbuf[i] = get_normalized_hemisphere_buffer_angles(0, ((float)i) / phi_o_size).x;
    }
    phioBuffer->unmap();
    optix::Buffer thetaoBuffer = create_buffer<float>(mContext, RT_BUFFER_INPUT, theta_o_size);
    float * tbuf = reinterpret_cast<float*>(thetaoBuffer->map());
    for(int i = 0; i < theta_o_size; i++)
    {
        tbuf[i] = get_normalized_hemisphere_buffer_angles(((float)i) / theta_o_size, 0).y;
    }
    thetaoBuffer->unmap();
    mParameterBuffers.buffers[4] = (phioBuffer)->getId();
    mParameterBuffers.buffers[3] = (thetaoBuffer)->getId();

    mManager = std::make_unique<BSSRDFParameterManager>(parameters);
    mInitialized = true;
}

EmpiricalBSSRDF::EmpiricalBSSRDF(optix::Context & context): BSSRDF(context, ScatteringDipole::EMPIRICAL_BSSRDF)
{
	std::string ptx_path_output = get_path_ptx("empirical_bssrdf.cu");
    //static optix::Program p = context->createProgramFromPTXFile(ptx_path_output, "eval_empirical_bssrdf");
    //mContext["empirical_bssrdf"]->setProgramId(p);

    mDataBuffers.buffers[0] = 0;
    mDataBuffers.buffers[1] = 0;
    mDataBuffers.buffers[2] = 0;
    mDataBuffers.test = 1;

    mBSSRDFFile = Folders::data_folder + "/bssrdf/test.bssrdf";
}

void EmpiricalBSSRDF::load(const ScatteringMaterialProperties & props)
{
    if(!mInitialized)
    {
        Logger::info << "Loading new material for empirical bssrdf..." << std::endl;
        prepare_buffers();

        mContext["empirical_bssrdf_parameters"]->setUserData(sizeof(EmpiricalParameterBuffer), &mParameterBuffers);
        auto id1 = rtBufferId<int>(mParameterSizeBuffer->getId());
        mContext["empirical_bssrdf_parameters_size"]->setUserData(sizeof(rtBufferId<int>), &id1);
        mDataBuffers.test = 50;

        std::vector<size_t> state;
        mManager->get_material_index(0.2, 0.0f, 1.1f, state);

        for(int i = 0; i < 3; i++)
        {
            optix::Buffer buf = mContext->getBufferFromId(mDataBuffers.buffers[i].getId());
            float * data = reinterpret_cast<float*>(buf->map());
            mBSSRDFLoader->load_material_slice(data, state);
            buf->unmap();

            RTsize w;
            buf->getSize(w);
            mContext["empirical_buffer_size"]->setInt(w);
        }

        mContext["empirical_buffer"]->setUserData(sizeof(EmpiricalDataBuffer), &mDataBuffers);
        Logger::info << "Loading complete." << std::endl;
    }
}