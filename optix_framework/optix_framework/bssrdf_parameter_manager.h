#pragma once
#include <map>
#include <algorithm>
#include <vector>
#include <string>
#include "logger.h"
#include "host_device_common.h"
#include "empirical_bssrdf_common.h"

#define INVALID_INDEX ((size_t)(-1))

// Indices defining the dimensions in the BSSRDF storage
// First two dimensions depend on the type of renderer: (x,y) for plane, (theta_o, phi_o) for hemisphere.
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

/*
 * Multi dimensional index. Can be incremented mUp to a maximum. We use it so simulate increasing indices on a simulation.
 */
class ParameterState
{
public:
	ParameterState() {}
	ParameterState(const std::vector<size_t>& data) : mData(data) {}

	std::string tostring();
	const size_t& operator[](const size_t& idx) const;
	size_t& operator[](const size_t& idx);
	bool operator==(const ParameterState& b) const;
	size_t* data();
    std::vector<size_t> mData;
};

/*
 * Stores a collection of parameters for a BSSRDF rendering. Allows to advance the state of a running bssrdf simulation, and get the relative parameters for that simulation.
 */
class BSSRDFParameterManager
{
public:
	BSSRDFParameterManager() : parameters(original_parameters) {}
	BSSRDFParameterManager(std::map<size_t, std::vector<float>> p) : parameters(p) {}
	std::map<size_t, std::vector<float>> parameters;
	static std::map<size_t, std::vector<float>> original_parameters;

	size_t size() const;
	void get_parameters(const ParameterState& state, float& theta_i, optix::float2& r, optix::float2& theta_s, float& albedo, float& g, float& eta);
	bool get_single_index(const float val, const size_t idx, size_t& idx_res);
	bool get_index(const float theta_i, const float r, const float theta_s, const float albedo, const float g,
	               const float eta, std::vector<size_t>& state);
	bool get_material_index(const float albedo, const float g, const float eta, std::vector<size_t>& state);
	ParameterState next(const ParameterState& state);
	bool is_valid(const ParameterState& state);
	size_t get_size();
	std::vector<size_t> get_dimensions() const;
	static const std::map<size_t, std::string>& get_parameter_names(OutputShape::Type type);;

private:
	bool increment(size_t src, size_t size, size_t& dst);

	static const ParameterState invalid_index;
	static std::map<size_t, std::string> parameter_names_hemi;
	static std::map<size_t, std::string> parameter_names_plane;
};