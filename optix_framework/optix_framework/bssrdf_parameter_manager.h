#pragma once
#include <map>
#include <algorithm>
#include <vector>
#include <string>
#include "logger.h"
#include "host_device_common.h"
#include "empirical_bssrdf_common.h"

#define INVALID_INDEX ((size_t)(-1))

#define dim_2_index 7
#define dim_1_index 6
#define theta_i_index 5
#define r_index 4
#define theta_s_index 3
#define albedo_index 2
#define g_index 1
#define eta_index 0

size_t flatten_index(const std::vector<size_t> & idx, const std::vector<size_t> & size);
std::vector<size_t> unravel_index(const size_t& idx, const std::vector<size_t>& size);

struct ParameterState
{
	ParameterState() {}
	ParameterState(const std::vector<size_t>& data) : mData(data) {}

	std::string tostring()
	{
		std::string res = "(";
		for (int i = 0; i < mData.size(); i++)
		{
			res += std::to_string(mData[i]) + ((i == mData.size() - 1) ? "" : " ");
		}
		return res + ")";
	}

	const size_t& operator[](const size_t & idx) const
	{
		if (idx >= 8)
			Logger::error << "Out of bounds!" << std::endl;
		return mData[idx];
	}

	size_t& operator[](const size_t & idx)
	{
		if (idx >= mData.size())
			Logger::error << "Out of bounds!" << std::endl;
		return mData[idx];
	}

	bool operator==(const ParameterState &b) const
	{
		bool equal = true;
		for (int i = 0; i < mData.size(); i++)
		{
			equal &= b.mData[i] == mData[i];
		}
		return equal;
	}

	std::vector<size_t> mData;
};

class BSSRDFParameterManager
{
public:
	BSSRDFParameterManager() : parameters(original_parameters) {}
	BSSRDFParameterManager(std::map<size_t, std::vector<float>> p) : parameters(p) {}
	std::map<size_t, std::vector<float>> parameters;

	static std::map<size_t, std::vector<float>> original_parameters;
	size_t size() const
	{
		std::vector<size_t> dims = get_dimensions();
		size_t res = 1;
		for (const size_t& i : dims)
			res *= i;
		return res;
	}

	void get_parameters(const ParameterState & state, float & theta_i, optix::float2 & r, optix::float2 & theta_s, float & albedo, float & g, float & eta)
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

	bool get_single_index(const float val, const size_t idx, size_t & idx_res)
	{
		auto res = std::find_if(std::begin(parameters[idx]), std::end(parameters[idx]), [val](float & f)->bool { return val - f < 1e-6f; });
		if (res == std::end(parameters[idx]))
			return false;
		idx_res = std::distance(std::begin(parameters[idx]), res);
		return true;
	}

	bool get_index(const float theta_i, const float r, const float theta_s, const float albedo, const float g, const float eta, std::vector<size_t> & state)
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

	bool get_material_index(const float albedo, const float g, const float eta, std::vector<size_t> & state)
	{
		bool success = true;
		state.resize(3);
		success &= get_single_index(albedo, albedo_index, state[albedo_index]);
		success &= get_single_index(g, g_index, state[g_index]);
		success &= get_single_index(eta, eta_index, state[eta_index]);
		return success;
	}

	ParameterState next(const ParameterState & state)
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
	  
	bool is_valid(const ParameterState & state)
	{
		return !(state == invalid_index);
	}

	size_t get_size()
	{
		size_t r = 1;
		for (auto c : get_dimensions())
		{
			r *= c;
		}
		return r;
	}

	std::vector<size_t> get_dimensions() const
	{
		std::vector<size_t> dims(parameters.size());
		for (int i = 0; i < dims.size(); i++)
			dims[i] = parameters.at(i).size();
		dims[r_index] -= 1; 		// This span deltas!
		dims[theta_s_index] -= 1; 	// This span deltas!
		return dims;
	}


    static const std::map<size_t, std::string>& get_parameter_names(OutputShape::Type type)
    {
        return type == OutputShape::HEMISPHERE? parameter_names_hemi : parameter_names_plane;
    };

private:
	bool increment(size_t src, size_t size, size_t & dst)
	{
		dst = (src + 1) % size;
		return ((src + 1) / size) >= 1;
	}

	static const ParameterState invalid_index;
	static std::map<size_t, std::string> parameter_names_hemi;
	static std::map<size_t, std::string> parameter_names_plane;

};