#include "bssrdf_parameter_manager.h"
#include "assert.h"

const ParameterState BSSRDFParameterManager::invalid_index = ParameterState({INVALID_INDEX, INVALID_INDEX,
                                                                             INVALID_INDEX, INVALID_INDEX,
                                                                             INVALID_INDEX, INVALID_INDEX});;

#define stringify_pair(x) x, #x
std::map<size_t, std::string> BSSRDFParameterManager::parameter_names_hemi = {
        {dim_2_index, "theta_o"},
        {dim_1_index, "phi_o"},
        {stringify_pair(theta_i_index)},
        {stringify_pair(r_index)},
        {stringify_pair(theta_s_index)},
        {stringify_pair(albedo_index)},
        {stringify_pair(g_index)},
        {stringify_pair(eta_index)}
};

std::map<size_t, std::string> BSSRDFParameterManager::parameter_names_plane = {
        {dim_2_index, "y"},
        {dim_1_index, "x"},
        {stringify_pair(theta_i_index)},
        {stringify_pair(r_index)},
        {stringify_pair(theta_s_index)},
        {stringify_pair(albedo_index)},
        {stringify_pair(g_index)},
        {stringify_pair(eta_index)}
};

#undef stringify_pair

std::map<size_t, std::vector<float>> BSSRDFParameterManager::original_parameters = {
        {theta_i_index, {0,     15,    30,    45,    60,   70,   80,   88}},
        {r_index,       {0.0f,  0.05f, 0.1f,  0.2f,  0.4f, 0.6f, 0.8f, 1.0f, 2.0f, 4.0f,  8.0f, 10.0f}},
        {theta_s_index, {0,     15,    30,    45,    60,   75,   90,   105,  120,  135,   150,  165, 180}},
        {albedo_index,  {0.01f, 0.1f,  0.2f,  0.3f,  0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f,  0.99f}},
        {g_index,       {-0.9f, -0.7f, -0.5f, -0.3f, 0.0f, 0.3f, 0.5f, 0.7f, 0.9f, 0.95f, 0.99f}},
        {eta_index,     {1.0f,  1.1f,  1.2f,  1.3f,  1.4f}}
};


size_t flatten_index(const std::vector<size_t> &idx, const std::vector<size_t> &size)
{
    size_t id = idx[eta_index];
    for (int i = 1; i < size.size(); i++)
    {
        assert(size[i] > idx[i]);
        id = id * size[i] + idx[i];
    }
    return id;
}

std::vector<size_t> unravel_index(const size_t &idx, const std::vector<size_t> &size)
{
    size_t index = idx;
    std::vector<size_t> res(size.size(), 0);
    for (int i = (int) size.size() - 1; i >= 0; i--)
    {
        res[i] = index % size[i];
        index = index / size[i];
    }
    return res;
}
