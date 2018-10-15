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
#define INVALID_INDEX ((size_t)(-1))

// Indices defining the dimensions in the BSSRDF storage
// First two dimensions depend on the type of renderer: (x,y) for plane, (theta_o, phi_o) for hemisphere.
#define number_dimensions 8
#define dim_2_index 7
#define dim_1_index 6
#define theta_i_index 5
#define r_index 4
#define theta_s_index 3
#define albedo_index 2
#define g_index 1
#define eta_index 0

#define USE_SMALL_FILES

using IndexType = size_t;

/*
 * Multi dimensional index. Can be incremented mUp to a maximum. We use it so simulate increasing indices on a simulation.
 */
class EmpiricalBSSRDF;

class ParameterStateNew
{
public:
    friend class EmpiricalBSSRDF;
    ParameterStateNew(const std::vector<size_t>& data, const std::vector<size_t>& dims) : mData(data), mDimensions(dims)
    {
        assert(mData.size() == mDimensions.size());
    }

    std::string tostring() const;
    const size_t& operator[](const size_t& idx) const;
    size_t& operator[](const size_t& idx);

    bool operator==(const ParameterStateNew& b) const;
    bool operator!=(const ParameterStateNew& b) const { return !(this->operator==(b)); }
    size_t* data();

    ParameterStateNew next(const ParameterStateNew& state);
    bool is_valid() const;
    std::vector<size_t> get_dimensions() const { return mDimensions;}
    size_t flatten();

    const static ParameterStateNew invalid_index;

private:
    bool increment(size_t src, size_t size, size_t& dst);
    std::vector<size_t> mData;
    std::vector<size_t> mDimensions;
};


class EmpiricalBSSRDF
{
public:
    EmpiricalBSSRDF();

    // Sets the filename for this BSSRDF.
    void set_filename(const std::string & filename);
    std::string get_filename() const;

    // Sets filename, also tries to load parameters
    bool load_header(const std::string & filename);
    void save_header(const std::string & filename);

    std::vector<size_t> get_material_dimensions() const;
    std::vector<size_t> get_material_geometry_dimensions() const;
    std::vector<size_t> get_material_geometry_hemisphere_dimensions() const;

    size_t get_material_combinations() const;
    size_t get_material_geometry_combinations() const;
    size_t get_size() const;

    OutputShape::Type get_shape() const;
    void set_shape(const OutputShape::Type & shape);

    size_t get_material_slice_size();
    size_t get_hemisphere_size();

    std::map<size_t, std::vector<float>> get_parameters_copy();
    void set_parameters(const std::map<IndexType, std::vector<float>>& parameters);
    void set_parameter_values(const IndexType& index, const std::vector<float>& parameters);

    bool load_material_slice(float * bssrdf_data, const std::vector<IndexType> & idx);
    bool load_hemisphere(float * bssrdf, const ParameterStateNew & idx);
    void set_material_slice(const float * bssrdf_data, const std::vector<IndexType> & idx);
    void set_hemisphere(const float * bssrdf, const ParameterStateNew & idx);

    size_t get_dimension_2() { return mDimensions[dim_2_index]; }
    size_t get_dimension_1() { return mDimensions[dim_1_index]; }

    static const std::map<size_t, std::vector<float>> get_default_parameters();

    void get_parameters(const ParameterStateNew& state, float& theta_i, optix::float2& r, optix::float2& theta_s, float& albedo, float& g, float& eta);
    bool get_single_index(const float val, const size_t idx, size_t& idx_res);
    bool get_index(const float theta_i, const float r, const float theta_s, const float albedo, const float g,
                   const float eta, ParameterStateNew& state);

    bool get_material_index(const float albedo, const float g, const float eta, std::vector<size_t>& state);
    static const std::map<size_t, std::string>& get_parameter_names(OutputShape::Type type);;

    void set_hemisphere_size(size_t h1, size_t h2);
    void reset_parameters();

    ParameterStateNew begin_index() const;

private:
    IndexType mDimensions[number_dimensions];
    std::map<IndexType, std::vector<float>> mParameters;
    OutputShape::Type mShape;
    std::string mFilename = "";

    bool parse_header();
    std::string create_header();
    void recalculate_dimensions();

    static std::map<size_t, std::vector<float>> default_parameters;
    static std::map<size_t, std::string> parameter_names_hemi;
    static std::map<size_t, std::string> parameter_names_plane;

};
