#include "neural_network_sampler.h"
#include "logger.h"
#include "optix_utils.h"
#include <Eigen/Dense>
#include <fstream>

using Eigen::MatrixXf;

bool exists(const char *fileName)
{
    // Check if a file exists
    std::ifstream infile(fileName);
    return infile.good();
}

NeuralNetworkSampler::NeuralNetworkSampler(optix::Context & ctx) : mContext(ctx)
{
    // Retrieve hypernetwork parameters
    std::string filepath(Folders::data_folder);
    std::string filename("parameters.bin");
    std::string network_path = filepath + filename;

    // Check if the trained network file exists or not
    if (exists(network_path.c_str()))
    {
        // Neural Network trained file format (binary)
        // Contents:
        //     n (number of hypernetwork parameters)

        // For each layer there will be corresponding weights and biases.
        //     in_size out_size (weights' dimensions)
        //     weights

        //     in_size out_size (biases' dimensions)
        //     biases
        FILE *fp = fopen(network_path.c_str(), "rb");
        int params_size;
        int num_layers;

        // Get the number of layers
        fread(&params_size, sizeof(int), 1, fp);
        num_layers = int(params_size / 2);

        // Create place holders for hypernetwork parameters
        MatrixXf *weights = new MatrixXf[num_layers];
        MatrixXf *biases = new MatrixXf[num_layers];

        // Allocate storage for OptiX buffers
        mHyperNetworkWeights.resize(num_layers);
        mHyperNetworkBiases.resize(num_layers);
        mContext["hypernetwork_num_layers"]->setInt(num_layers);

        // Read hypernetwork parameters
        for (int i = 0, layer_index = 0; i < params_size; ++i)
        {
            // Retrieve parameters' dimensions
            int in_size;
            int out_size;
            int total_size;
            char name[1024];

            fread(&in_size, sizeof(int), 1, fp);
            fread(&out_size, sizeof(int), 1, fp);
            total_size = in_size * out_size;

            if (i % 2 == 0)
            {
                // Initialize weights
                MatrixXf &weight = weights[layer_index];
                weight = MatrixXf::Zero(out_size, in_size);

                fread(weight.data(), sizeof(float), total_size, fp);

                // Create OptiX buffers
                mHyperNetworkWeights[layer_index] = mContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, total_size);
                memcpy(mHyperNetworkWeights[layer_index]->map(), weight.transpose().data(), sizeof(float) * total_size);
                mHyperNetworkWeights[layer_index]->unmap();

                // Send data to GPU
                sprintf(name, "hypernetwork_layer%d_weights", layer_index + 1);
                mContext[std::string(name)]->setBuffer(mHyperNetworkWeights[layer_index]);

                sprintf(name, "hypernetwork_layer%d_in_size", layer_index + 1);
                mContext[std::string(name)]->setInt(in_size);

                sprintf(name, "hypernetwork_layer%d_out_size", layer_index + 1);
                mContext[std::string(name)]->setInt(out_size);
            }
            else
            {
                // Assign biases
                MatrixXf &bias = biases[layer_index];
                bias = MatrixXf::Zero(in_size, out_size);

                fread(bias.data(), sizeof(float), total_size, fp);

                // Create OptiX buffers
                mHyperNetworkBiases[layer_index] = mContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, total_size);
                memcpy(mHyperNetworkBiases[layer_index]->map(), bias.data(), sizeof(float) * total_size);
                mHyperNetworkBiases[layer_index]->unmap();

                // Send to GPU
                sprintf(name, "hypernetwork_layer%d_biases", layer_index + 1);
                mContext[std::string(name)]->setBuffer(mHyperNetworkBiases[layer_index]);

                ++layer_index;
            }
        }
        fclose(fp);
    }

    std::string integral_filename("integrals.bin");
    std::string integral_path = filepath + integral_filename;

    // Check if the integrals exist or not
    if (exists(integral_path.c_str())) {
        // Reading the dimensions of the array first
        FILE *fp = fopen(integral_path.c_str(), "rb");
        int dimensionality;
        fread(&dimensionality, sizeof(int), 1, fp);
        assert(dimensionality == 4);
        // Now we read the dimensionality of each size of the array.
        int * dimensions = new int[dimensionality];
        fread(&dimensions[0], sizeof(int), 4, fp);

        //printf("%d %d %d %d\n", dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
        // We support only one dimension in eta, to allow a 3d texture for sampling.
        assert(dimensions[0] == 1);

        RTsize buffer_dims[3];
        size_t total_size = 1;
        for(int i = 0; i < 3; i++)
        {
            buffer_dims[i] = dimensions[i+1];
            total_size *= dimensions[i+1];
        }

        // Creating buffer to hold the data.
        optix::Buffer integral_data = ctx->createBuffer(RT_BUFFER_INPUT);
        integral_data->setFormat(RT_FORMAT_FLOAT);
        integral_data->setSize(3, &buffer_dims[0]);
        float* data = (float*)integral_data->map();
        fread(&data, sizeof(float), total_size, fp);

        // Creating texture.
        optix::TextureSampler texture = ctx->createTextureSampler();
        texture->setBuffer(integral_data);
        texture->setFilteringModes(RT_FILTER_LINEAR,RT_FILTER_LINEAR,RT_FILTER_NONE);
        texture->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        texture->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
        texture->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
        texture->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
        texture->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
        ctx["integral_texture"]->setInt(texture->getId());

        fclose(fp);
    }
}

bool NeuralNetworkSampler::on_draw()
{
    return false; // Nothing has changed.
}

void NeuralNetworkSampler::load(float relative_ior, const ScatteringMaterialProperties & props)
{
    //Logger::info << "Log something..." << std::endl;
    // It will be called the next frame, if something changes.
    // You can trigger this by returning true in the on_draw method. 
    // Here you can do context->set stuff, etc.
}
