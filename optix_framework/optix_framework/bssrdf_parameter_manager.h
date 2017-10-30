#pragma once
#include <map>
#include <algorithm>
#include <vector>
#include <string>
#include "logger.h"

#define INVALID_INDEX ((size_t)(-1))

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

	std::map<size_t, std::vector<float>> parameters = {
		{ theta_i_index,{ 0, 15, 30, 45, 60, 70, 80, 88 } },
		{ r_index,{ 0.01f, 0.05f, 0.1f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 2.0f, 4.0f, 8.0f, 10.0f } },
		{ theta_s_index,{ 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180 } },
		{ albedo_index,{ 0.01f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.99f } },
		{ g_index,{ -0.9f, -0.7f, -0.5f, -0.3f, 0.0f, 0.3f, 0.5f, 0.7f, 0.9f, 0.95f, 0.99f } },
		{ eta_index,{ 1.0f, 1.1f, 1.2f, 1.3f, 1.4f } }
	};

	static std::map<size_t, std::string> parameter_names;

	size_t size() const
	{
		std::vector<size_t> dims = get_dimensions();
		size_t res = 1;
		for (const size_t& i : dims)
			res *= i;
		return res;
	}

	void get_parameters(const ParameterState & state, float & theta_i, float & r, float & theta_s, float & albedo, float & g, float & eta)
	{
		if (!is_valid(state))
			return;
		theta_i = parameters[theta_i_index][state[theta_i_index]];
		r = parameters[r_index][state[r_index]];
		theta_s = parameters[theta_s_index][state[theta_s_index]];
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

	bool get_index(const float theta_i, const float r, const float theta_s, const float albedo, const float g, const float eta, ParameterState & state)
	{
		bool success = true;
		success &= get_single_index(theta_i, theta_i_index, state.mData[theta_i_index]);
		success &= get_single_index(r, r_index, state.mData[r_index]);
		success &= get_single_index(theta_s, theta_s_index, state.mData[theta_s_index]);
		success &= get_single_index(albedo, albedo_index, state.mData[albedo_index]);
		success &= get_single_index(g, g_index, state.mData[g_index]);
		success &= get_single_index(eta, eta_index, state.mData[eta_index]);
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

	std::vector<size_t> BSSRDFParameterManager::get_dimensions() const
	{
		std::vector<size_t> dims(parameters.size());
		for (int i = 0; i < dims.size(); i++)
			dims[i] = parameters.at(i).size();
		return dims;
	}

private:
	bool increment(size_t src, size_t size, size_t & dst)
	{
		dst = (src + 1) % size;
		return ((src + 1) / size) >= 1;
	}

	static const ParameterState invalid_index; 
};