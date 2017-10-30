#include "bssrdf_parameter_manager.h"
#include "assert.h"

const ParameterState BSSRDFParameterManager::invalid_index = ParameterState({ INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX });;

#define stringify_pair(x) x, #x
std::map<size_t, std::string> BSSRDFParameterManager::parameter_names = {
	{ stringify_pair(theta_i_index) },
	{ stringify_pair(r_index) },
	{ stringify_pair(theta_s_index) },
	{ stringify_pair(albedo_index) },
	{ stringify_pair(g_index) },
	{ stringify_pair(eta_index) }
};
#undef stringify_pair


size_t flatten_index(const std::vector<size_t>& idx, const std::vector<size_t>& size)
{
	size_t id = idx[eta_index];
	for (int i = 1; i < size.size(); i++)
	{
		assert(size[i] > idx[i]);
		id = id * size[i] + idx[i];
	}
	return id;
}

std::vector<size_t> unravel_index(const size_t& idx, const std::vector<size_t>& size)
{
	size_t index = idx;
	std::vector<size_t> res(size.size(), 0);
	for (int i = (int)size.size() - 1; i >= 0; i--)
	{
		res[i] = index % size[i];
		index = index / size[i];
	}
	return res;
}
