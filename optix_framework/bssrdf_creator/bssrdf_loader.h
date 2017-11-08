#pragma once
#include <string>
#include <vector>
#include <map>
#include "bssrdf_parameter_manager.h"

#define USE_SMALL_FILES

#define bssrdf_delimiter std::string("BSSRDF")
#define size_delimiter std::string("SIZE")
#define parameter_delimiter std::string("PARAMETER")


class BSSRDFLoader
{
public:
	BSSRDFLoader(const std::string & filename);
	void get_dimensions(std::vector<size_t> & dimensions);
	size_t get_material_slice_size();
	size_t get_hemisphere_size();
	const std::map<size_t, std::vector<float>>& get_parameters();
	bool load_material_slice(float * bssrdf_data, const std::vector<size_t> & idx);
	bool load_hemisphere(float * bssrdf, const std::vector<size_t> & idx);
	size_t get_hemisphere_theta_o() { return mThetaoSize; }
	size_t get_hemisphere_phi_o() { return mPhioSize; }

private:
	bool parse_header();
	std::vector<size_t> mDimensions;
	size_t mBSSRDFStart = 0;
	std::string mFileName;
	std::map<size_t, std::vector<float>> mParameters;
	size_t mThetaoSize, mPhioSize;
};

class BSSRDFExporter
{
public:
	BSSRDFExporter(const std::string & filename, const BSSRDFParameterManager & manager, size_t size_theta_o, size_t size_phi_o);
	size_t get_material_slice_size();
	size_t get_hemisphere_size();

	void set_material_slice(const float * bssrdf_data, const std::vector<size_t> & idx);
	void set_hemisphere(const float * bssrdf, const std::vector<size_t> & idx);

private:
	std::string create_header();
	std::string mHeader;
	std::string mFileName;
	size_t mThetaoSize, mPhioSize;
	
	const BSSRDFParameterManager & mManager;
};