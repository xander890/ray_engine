#include "neural_network_sampler.h"
#include "logger.h"
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
