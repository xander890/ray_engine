#pragma once
#include "bssrdf_parameter_manager.h"
#include "empirical_bssrdf_common.h"
#include <string>
#include <vector>
#include <map>

/*
 * Utility class to import and export BSSRDF empirical data. This is largley our defined format. Format prepares a common plain text file (.bssrdf) that includes info on bins, parameters used, etc. Then, a certain number of files is generated, one for each (eta, albedo, g) contribution.
 *
 */
#define USE_SMALL_FILES

#define bssrdf_delimiter std::string("BSSRDF")
#define size_delimiter std::string("SIZE")
#define parameter_delimiter std::string("PARAMETER")
#define shape_delimiter std::string("SHAPE")

/*
 * Imports BSSRDF data.
 */
class BSSRDFImporter
{
public:
	// Provide path to the main .bssrdf file.
	BSSRDFImporter(const std::string & filename);

	~BSSRDFImporter() = default;

	void get_dimensions(std::vector<size_t> & dimensions);
	size_t get_material_slice_size();
	size_t get_hemisphere_size();
	const std::map<size_t, std::vector<float>>& get_parameters();
	bool load_material_slice(float * bssrdf_data, const std::vector<size_t> & idx);
	bool load_hemisphere(float * bssrdf, const std::vector<size_t> & idx);
	size_t get_hemisphere_dimension_2() { return mDimensions[dim_2_index]; }
	size_t get_hemisphere_dimension_1() { return mDimensions[dim_1_index]; }
	OutputShape::Type get_shape() { return mOutputShape; }

private:
	bool parse_header();
	std::vector<size_t> mDimensions;
	std::string mFileName;
	std::map<size_t, std::vector<float>> mParameters;
	OutputShape::Type mOutputShape;
};

/*
* Exports BSSRDF data.
*/
class BSSRDFExporter
{
public:
	BSSRDFExporter(const OutputShape::Type shape, const std::string & filename, const BSSRDFParameterManager & manager, size_t size_theta_o, size_t size_phi_o);
	size_t get_material_slice_size();
	size_t get_hemisphere_size();

	void set_material_slice(const float * bssrdf_data, const std::vector<size_t> & idx);
	void set_hemisphere(const float * bssrdf, const std::vector<size_t> & idx);

private:
	std::string create_header();
	std::string mHeader;
	std::string mFileName;
	size_t mDim2Size, mDim1Size;
	OutputShape::Type mOutputShape;

	const BSSRDFParameterManager & mManager;
};