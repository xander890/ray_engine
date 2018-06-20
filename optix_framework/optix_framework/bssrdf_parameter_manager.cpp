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
        {dim_2_index, "x"},
        {dim_1_index, "y"},
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

std::string ParameterState::tostring()
{
	std::string res = "(";
	for (int i = 0; i < mData.size(); i++)
	{
		res += std::to_string(mData[i]) + ((i == mData.size() - 1) ? "" : " ");
	}
	return res + ")";
}

const size_t& ParameterState::operator[](const size_t& idx) const
{
	if (idx >= 8)
		Logger::error << "Out of bounds!" << std::endl;
	return mData[idx];
}

size_t& ParameterState::operator[](const size_t& idx)
{
	if (idx >= mData.size())
		Logger::error << "Out of bounds!" << std::endl;
	return mData[idx];
}

bool ParameterState::operator==(const ParameterState& b) const
{
	bool equal = true;
	for (int i = 0; i < mData.size(); i++)
	{
		equal &= b.mData[i] == mData[i];
	}
	return equal;
}

size_t BSSRDFParameterManager::size() const
{
	std::vector<size_t> dims = get_dimensions();
	size_t res = 1;
	for (const size_t& i : dims)
		res *= i;
	return res;
}

void BSSRDFParameterManager::get_parameters(const ParameterState& state, float& theta_i, optix::float2& r,
                                            optix::float2& theta_s, float& albedo, float& g, float& eta)
{
	if (!is_valid(state))
		return;
	theta_i = parameters[theta_i_index][state[theta_i_index]];
	r.x = parameters[r_index][state[r_index]];
	theta_s.x = parameters[theta_s_index][state[theta_s_index]];
	r.y = parameters[r_index][state[r_index] + 1];
	theta_s.y = parameters[theta_s_index][state[theta_s_index] + 1];
	albedo = parameters[albedo_index][state[albedo_index]];
	g = parameters[g_index][state[g_index]];
	eta = parameters[eta_index][state[eta_index]];
}

bool BSSRDFParameterManager::get_single_index(const float val, const size_t idx, size_t& idx_res)
{
	auto res = std::find_if(std::begin(parameters[idx]), std::end(parameters[idx]),
	                        [val](float& f)-> bool { return val - f < 1e-6f; });
	if (res == std::end(parameters[idx]))
		return false;
	idx_res = std::distance(std::begin(parameters[idx]), res);
	return true;
}

bool BSSRDFParameterManager::get_index(const float theta_i, const float r, const float theta_s, const float albedo,
                                       const float g, const float eta, std::vector<size_t>& state)
{
	bool success = true;
	state.resize(6);
	success &= get_single_index(theta_i, theta_i_index, state[theta_i_index]);
	success &= get_single_index(r, r_index, state[r_index]);
	success &= get_single_index(theta_s, theta_s_index, state[theta_s_index]);
	success &= get_single_index(albedo, albedo_index, state[albedo_index]);
	success &= get_single_index(g, g_index, state[g_index]);
	success &= get_single_index(eta, eta_index, state[eta_index]);
	return success;
}

bool BSSRDFParameterManager::get_material_index(const float albedo, const float g, const float eta,
                                                std::vector<size_t>& state)
{
	bool success = true;
	state.resize(3);
	success &= get_single_index(albedo, albedo_index, state[albedo_index]);
	success &= get_single_index(g, g_index, state[g_index]);
	success &= get_single_index(eta, eta_index, state[eta_index]);
	return success;
}

ParameterState BSSRDFParameterManager::next(const ParameterState& state)
{
	ParameterState val = state;
	std::vector<size_t> dims = get_dimensions();
	int i;
	for (i = (int)dims.size() - 1; i >= 0; i--)
	{
		// increment returns true if overflow, so we keep adding.
		if (!increment(state[i], dims[i], val[i]))
		{
			break;
		}
	}

	// When the last index overflows.
	if (i == -1)
	{
		return invalid_index;
	}
	return val;
}

bool BSSRDFParameterManager::is_valid(const ParameterState& state)
{
	return !(state == invalid_index);
}

size_t BSSRDFParameterManager::get_size()
{
	size_t r = 1;
	for (auto c : get_dimensions())
	{
		r *= c;
	}
	return r;
}

std::vector<size_t> BSSRDFParameterManager::get_dimensions() const
{
	std::vector<size_t> dims(parameters.size());
	for (int i = 0; i < dims.size(); i++)
		dims[i] = parameters.at(i).size();
	dims[r_index] -= 1; // This span deltas!
	dims[theta_s_index] -= 1; // This span deltas!
	return dims;
}

const std::map<size_t, std::string>& BSSRDFParameterManager::get_parameter_names(OutputShape::Type type)
{
	return type == OutputShape::HEMISPHERE ? parameter_names_hemi : parameter_names_plane;
}

bool BSSRDFParameterManager::increment(size_t src, size_t size, size_t& dst)
{
	dst = (src + 1) % size;
	return ((src + 1) / size) >= 1;
}
