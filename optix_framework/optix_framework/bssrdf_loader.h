#pragma once
#include <string>
#include <vector>
#include <map>

#define USE_SMALL_FILES

#define bssrdf_delimiter std::string("BSSRDF")
#define size_delimiter std::string("SIZE")
#define parameter_delimiter std::string("PARAMETER")

#define	theta_o_index 7
#define phi_o_index 6
#define theta_i_index 5
#define r_index 4
#define theta_s_index 3
#define albedo_index 2
#define g_index 1
#define eta_index 0

size_t flatten_index(const std::vector<size_t> & idx, const std::vector<size_t> & size);
std::vector<size_t> unravel_index(const size_t& idx, const std::vector<size_t>& size);

class BSSRDFLoader
{
public:
	BSSRDFLoader(const std::string & filename, const std::map<size_t, std::string> & names);
	void get_dimensions(std::vector<size_t> & dimensions);
	size_t get_material_slice_size();
	size_t get_hemisphere_size();
	const std::map<size_t, std::vector<float>>& get_parameters();
	void load_material_slice(float * bssrdf_data, const std::vector<size_t> & idx);
	void load_hemisphere(float * bssrdf, const std::vector<size_t> & idx);

private:
	bool parse_header(const std::map<size_t, std::string> & names);
	std::vector<size_t> mDimensions;
	size_t mBSSRDFStart = 0;
	std::string mFileName;
	std::map<size_t, std::vector<float>> mParameters;
};

class BSSRDFExporter
{
public:
	BSSRDFExporter(const std::string & filename, const std::vector<size_t> & dimensions, const std::map<size_t, std::vector<float>> & parameters, const std::map<size_t, std::string> & names);
	size_t get_material_slice_size();
	size_t get_hemisphere_size();

	void set_material_slice(const float * bssrdf_data, const std::vector<size_t> & idx);
	void set_hemisphere(const float * bssrdf, const std::vector<size_t> & idx);

private:
	std::string create_header();
	std::vector<size_t> mDimensions;
	std::string mHeader;
	std::string mFileName;
	const std::map<size_t, std::vector<float>> & mParameters;
	const std::map<size_t, std::string> & mNames;
};