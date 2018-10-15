#include "empirical_bssrdf.h"
#include <fstream>
#include <sstream>
#include "logger.h"
#include "parsing_utils.h"
#include <algorithm>
#include <assert.h>
#include <string_utils.h>
#include "empirical_bssrdf_common.h"

#define bssrdf_delimiter std::string("BSSRDF")
#define size_delimiter std::string("SIZE")
#define parameter_delimiter std::string("PARAMETER")
#define shape_delimiter std::string("SHAPE")

const ParameterStateNew ParameterStateNew::invalid_index = ParameterStateNew({INVALID_INDEX, INVALID_INDEX,
                                                                             INVALID_INDEX, INVALID_INDEX,
                                                                             INVALID_INDEX, INVALID_INDEX}, {1,1,1,1,1,1});;

#define stringify_pair(x) x, #x
std::map<size_t, std::string> EmpiricalBSSRDF::parameter_names_hemi = {
        {dim_2_index, "theta_o"},
        {dim_1_index, "phi_o"},
        {stringify_pair(theta_i_index)},
        {stringify_pair(r_index)},
        {stringify_pair(theta_s_index)},
        {stringify_pair(albedo_index)},
        {stringify_pair(g_index)},
        {stringify_pair(eta_index)}
};

std::map<size_t, std::string> EmpiricalBSSRDF::parameter_names_plane = {
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

std::map<size_t, std::vector<float>> EmpiricalBSSRDF::default_parameters = {
        {theta_i_index, {0,     15,    30,    45,    60,   70,   80,   88}},
        {r_index,       {0.0f,  0.05f, 0.1f,  0.2f,  0.4f, 0.6f, 0.8f, 1.0f, 2.0f, 4.0f,  8.0f, 10.0f}},
        {theta_s_index, {0,     15,    30,    45,    60,   75,   90,   105,  120,  135,   150,  165, 180}},
        {albedo_index,  {0.01f, 0.1f,  0.2f,  0.3f,  0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f,  0.99f}},
        {g_index,       {-0.9f, -0.7f, -0.5f, -0.3f, 0.0f, 0.3f, 0.5f, 0.7f, 0.9f, 0.95f, 0.99f}},
        {eta_index,     {1.0f,  1.1f,  1.2f,  1.3f,  1.4f}}
};


size_t flatten_index(const std::vector<size_t> &idx, const std::vector<size_t> &size)
{
    assert(idx.size() == size.size());
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


std::vector<float> convert_to_rad(const std::vector<float> & vec)
{
    std::vector<float> n;
    for(auto & v : vec)
    {
        n.push_back(deg2rad(v));
    }
    return n;
}


std::string get_sub_filename(const std::string & filename, const std::vector<size_t> & idx, const std::map<size_t, std::vector<float>> & params)
{
	std::stringstream file;
	file << std::fixed;
	file.precision(2);
	std::string f, e;
	split_extension(filename, f, e);

	file << f << "_eta_" << params.at(eta_index)[idx[eta_index]] << "_g_" << params.at(g_index)[idx[g_index]] << "_alpha_" << params.at(albedo_index)[idx[albedo_index]] << e;
	std::string s = file.str();
	return s;
}

bool file_exists(const std::string & file)
{
	std::ifstream f(file);
	return (bool)f;
}

std::vector<float> get_parameter_x(int n)
{
    std::vector<float> res(n);
    for(int i = 0; i <= n; i++)
    {
        float normalized = static_cast<float>(i) / n;
        res[i] = normalized * 2 - 1;
    }
    return res;
}

std::vector<float> get_parameter_y(int n)
{
    std::vector<float> res(n);
    for(int i = 0; i <= n; i++)
    {
        float normalized = static_cast<float>(i) / n;
        res[i] = normalized * 2 - 1;
    }
    return res;
}

std::vector<float> get_parameter_theta(int n)
{
    std::vector<float> res(n);
    for(int i = 0; i <= n; i++)
    {
        float normalized = static_cast<float>(i) / n;
        float theta_o, phi_o;
        get_angles_polar(0, normalized, phi_o, theta_o);
        res[i] = theta_o;
    }
    return res;
}

std::vector<float> get_parameter_phi(int n)
{
    std::vector<float> res(n);
    for(int i = 0; i <= n; i++)
    {
        float normalized = static_cast<float>(i) / n;
        float theta_o, phi_o;
        get_angles_polar(normalized, 0, phi_o, theta_o);
        res[i] = phi_o;
    }
    return res;
}


void parameters_to_string(const std::map<size_t, std::vector<float>> & parameters, const std::map<size_t, std::string> & names, std::string & result)
{
	std::stringstream ss;
	for (auto & p : parameters)
	{
        std::string n = tostring(p.second);
        if(p.first == theta_i_index || p.first == theta_s_index)
        {
            n = tostring(convert_to_rad(p.second));
        }
		ss << parameter_delimiter << " " << names.at(p.first) << " " << n << std::endl;
	}
	result = ss.str();
}

bool EmpiricalBSSRDF::load_material_slice(float * bssrdf_data, const std::vector<size_t>& idx)
{
	if (idx.size() != 3)
	{
        Logger::error << "Material index is 3 dimensional." << std::endl;
		return false;
	}
#ifdef USE_SMALL_FILES
	size_t pos = 0;
	std::string filename = get_sub_filename(mFilename, idx, mParameters);
	if (!file_exists(filename))
	{
		Logger::error << "File not found. " << filename << std::endl;
		return false;
	}
	else
	{
		Logger::info << "Loading file: " << filename << std::endl;
	}
	std::ifstream ifs;
	ifs.open(filename, std::ofstream::in | std::ofstream::binary);
	ifs.seekg(pos);
	ifs.read(reinterpret_cast<char*>(bssrdf_data), get_material_slice_size() * sizeof(float));
	ifs.close();
#else
	size_t pos = flatten_index({ idx[0], idx[1], idx[2], 0, 0, 0, 0, 0 }, mDimensions) * sizeof(float);
	std::ifstream ifs;
	ifs.open(mFileName, std::ofstream::in | std::ofstream::binary);
	ifs.seekg(pos + mBSSRDFStart);
	ifs.read(reinterpret_cast<char*>(bssrdf_data), get_material_slice_size() * sizeof(float));
	ifs.close();
#endif
	return true;
}

bool EmpiricalBSSRDF::load_hemisphere(float * bssrdf_data, const ParameterStateNew& idx)
{
	if (idx.mData.size() != 6)
	{
		Logger::error << "Hemisphere index is 6 dimensional." << std::endl;
		return false;
	}
#ifdef USE_SMALL_FILES
	std::vector<size_t> dims = get_material_geometry_dimensions();
	dims[eta_index] = dims[albedo_index] = dims[g_index] = 1;
	size_t pos = flatten_index({ 0, 0, 0, idx[3], idx[4], idx[5] }, dims) * get_hemisphere_size() * sizeof(float);
	std::string s = get_sub_filename(mFilename, idx.mData, mParameters);
	if (!file_exists(s))
	{
		//Logger::error << "File not found. " << s << std::endl;
		return false;
	}
	else
	{
		Logger::info << "File found. Index " <<  flatten_index({ 0, 0, 0, idx[3], idx[4], idx[5] }, dims) << std::endl;
	}

	std::ifstream ifs(s, std::ofstream::in | std::ofstream::binary);
	ifs.seekg(pos);
	ifs.read(reinterpret_cast<char*>(bssrdf_data), get_hemisphere_size() * sizeof(float));
	ifs.close();
#else
	size_t pos = flatten_index({ idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], 0, 0 }, mDimensions) * sizeof(float);
	std::ifstream ifs;
	ifs.open(mFileName, std::ofstream::in | std::ofstream::binary);
	ifs.seekg(pos + mBSSRDFStart);
	ifs.read(reinterpret_cast<char*>(bssrdf_data), get_hemisphere_size() * sizeof(float));
	ifs.close();
#endif
	return true;
}
	
bool EmpiricalBSSRDF::parse_header()
{
	std::ifstream file(mFilename);
    std::string str;

	bool parsed_dimensions = false;
	bool shape_found = false;

	while (std::getline(file, str)) {
		if (str.size() >= size_delimiter.size() && str.substr(0, size_delimiter.size()).compare(size_delimiter) == 0)
		{
			std::stringstream ss (str.substr(size_delimiter.size()));
			for (int i = 0; i < number_dimensions; i++)
			{
				size_t size;
				ss >> size; 
				if (ss.fail())
					return false;
				mDimensions[i] = size;
			}
			parsed_dimensions = true;
		}
		if (str.size() >= shape_delimiter.size() && str.substr(0, shape_delimiter.size()).compare(shape_delimiter) == 0)
		{
			if(shape_found)
				Logger::warning << "Shape inconsistency found. Be careful to specify the shape before the parameters!" << std::endl;
			mShape = OutputShape::to_enum(str.substr(shape_delimiter.size()));
			shape_found = true;
		}
        if (str.size() >= parameter_delimiter.size() && str.substr(0, parameter_delimiter.size()) == parameter_delimiter)
        {
			if(!shape_found)
			{
				Logger::error << "Shape has not been found. Assuming hemisphere!" << std::endl;
                mShape = OutputShape::HEMISPHERE;
				shape_found = true;
			}
            std::stringstream ss(str.substr(parameter_delimiter.size()));
            std::string name;
            ss >> name;
            std::string s = ss.str();
            s.erase(0, name.length() + 1);
			auto names = get_parameter_names(mShape);
            auto res = std::find_if(std::begin(names), std::end(names), [&](const auto &pair)
            {
                return pair.second == name;
            });
            if (res != names.end())
                mParameters[res->first] = tovalue<std::vector<float>>(s);
        }
		if (!str.empty() && str[0] == '#')
			continue;
	}
	return parsed_dimensions;
}

size_t EmpiricalBSSRDF::get_material_slice_size()
{
	auto dims = get_material_geometry_hemisphere_dimensions();
	return get_hemisphere_size() * dims[theta_i_index] * dims[theta_s_index] * dims[r_index];
}

size_t EmpiricalBSSRDF::get_hemisphere_size()
{
	return get_dimension_1() * get_dimension_2();
}

void EmpiricalBSSRDF::set_material_slice(const float * bssrdf_data, const std::vector<size_t>& idx)
{
	if (idx.size() != 3)
		Logger::error << "Material index is 3 dimensional." << std::endl;
#ifdef USE_SMALL_FILES
	size_t pos = 0;
	std::ofstream ofs;
	std::string filename = get_sub_filename(mFilename, idx, mParameters);
	if (!file_exists(filename))
	{
		ofs.open(filename, std::ofstream::out);
		ofs.seekp(get_material_slice_size() * sizeof(float) - 1);
		ofs.put('\0');
		ofs.close();
	}
	ofs.seekp(pos);
	ofs.write(reinterpret_cast<const char*>(bssrdf_data), get_material_slice_size() * sizeof(float));
	ofs.close();
#else
	size_t pos = flatten_index({ idx[0], idx[1], idx[2], 0, 0, 0, 0, 0 }, mDimensions) * sizeof(float);
	std::ofstream ofs;
	ofs.open(mFileName, std::ofstream::in | std::ofstream::out | std::ofstream::binary);
	ofs.seekp(pos + mHeader.size());
	ofs.write(reinterpret_cast<const char*>(bssrdf_data), get_material_slice_size() * sizeof(float));
	ofs.close();
#endif
}

void EmpiricalBSSRDF::set_hemisphere(const float * bssrdf_data, const ParameterStateNew& idx)
{
	if (idx.mData.size() != 6)
		Logger::error << "Hemisphere index is 6 dimensional." << std::endl;
#ifdef USE_SMALL_FILES
	std::vector<size_t> dims = get_material_geometry_dimensions();
	dims[eta_index] = dims[albedo_index] = dims[g_index] = 1;
	size_t pos = flatten_index({ 0, 0, 0, idx[3], idx[4], idx[5]}, dims) * get_hemisphere_size() * sizeof(float);
	std::ofstream ofs;
	std::string filename = get_sub_filename(mFilename, idx.mData, mParameters);
	if (!file_exists(filename))
	{
		ofs.open(filename, std::ofstream::out);
		//ofs.seekp(get_material_slice_size() * sizeof(float) - 1);
		//ofs.put('\0');
		ofs.close();
	}
	ofs.open(filename, std::ofstream::in | std::ofstream::out | std::ofstream::binary);
	ofs.seekp(pos);
	ofs.write(reinterpret_cast<const char*>(bssrdf_data), get_hemisphere_size() * sizeof(float));
	ofs.close();
#else
	size_t pos = flatten_index({ idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], 0, 0 }, mDimensions) * sizeof(float);
	std::ofstream ofs;
	ofs.open(mFileName, std::ofstream::in | std::ofstream::out | std::ofstream::binary);
	ofs.seekp(pos + mHeader.size());
	ofs.write(reinterpret_cast<const char*>(bssrdf_data), get_hemisphere_size() * sizeof(float));
	ofs.close();
#endif
}

std::string EmpiricalBSSRDF::create_header()
{
	std::stringstream ss;
	ss << "# BSSRDF file format (version 0.2)" << std::endl;
	ss << "# Index dimensions is at follows:" << std::endl;
	ss << size_delimiter << " ";
	auto di = get_material_geometry_hemisphere_dimensions();
	for (int i = 0; i < di.size(); i++)
	{
		ss << di[i] << " ";
	}
	ss << std::endl;

	auto names = get_parameter_names(mShape);
	std::string dim_1_token = names[dim_1_index];
	std::string dim_2_token = names[dim_2_index];

	ss << "#";
    for(int i = 0; i < di.size(); i++)
    {
        ss << names[i] << " ";
    }

    ss << std::endl;
	ss << shape_delimiter << " " << OutputShape::to_string(mShape) << std::endl;

	std::string params;
	parameters_to_string(mParameters, names, params);
	ss << params;
    ss << bssrdf_delimiter << std::endl;
	return ss.str();
}

const std::map<size_t, std::vector<float>> EmpiricalBSSRDF::get_default_parameters() {
    return default_parameters;
}

void EmpiricalBSSRDF::save_header(const std::string &filename) {
    mFilename = filename;
    std::string header = create_header();
    std::ofstream of;
    of.open(filename, std::ofstream::out);
    of << header;
    of.close();
}

bool EmpiricalBSSRDF::load_header(const std::string &filename) {
    mFilename = filename;
    bool res = parse_header();
    if (!res)
    {
        Logger::error << "BSSRDF header parsing failed." << std::endl;
    }
    return res;
}

OutputShape::Type EmpiricalBSSRDF::get_shape() const {
    return mShape;
}

void EmpiricalBSSRDF::set_shape(const OutputShape::Type &shape) {
    mShape = shape;
    set_hemisphere_size(mDimensions[dim_1_index], mDimensions[dim_2_index]);
}

std::string EmpiricalBSSRDF::get_filename() const {
    return mFilename;
}

void EmpiricalBSSRDF::set_filename(const std::string &filename) {
    mFilename = filename;
}

std::map<size_t, std::vector<float>> EmpiricalBSSRDF::get_parameters_copy() {
    return mParameters;
}

void EmpiricalBSSRDF::set_parameters(const std::map<IndexType, std::vector<float>> &parameters) {
    mParameters = parameters;
    recalculate_dimensions();
}

void EmpiricalBSSRDF::set_parameter_values(const IndexType &index, const std::vector<float> &parameters) {
    assert(mParameters.count(index));
    mParameters[index] = parameters;
    recalculate_dimensions();
}

std::vector<size_t> EmpiricalBSSRDF::get_material_dimensions() const {
    std::vector<size_t> a;
    a.assign(mDimensions, mDimensions + 3);
    return a;
}

std::vector<size_t> EmpiricalBSSRDF::get_material_geometry_dimensions() const {
    std::vector<size_t> a;
    a.assign(mDimensions, mDimensions + 6);
    return a;
}

std::vector<size_t> EmpiricalBSSRDF::get_material_geometry_hemisphere_dimensions() const{
    std::vector<size_t> a;
    a.assign(mDimensions, mDimensions + 8);
    return a;
}


std::string ParameterStateNew::tostring() const
{
    std::string res = "(";
    for (int i = 0; i < mData.size(); i++)
    {
        res += std::to_string(mData[i]) + ((i == mData.size() - 1) ? "" : " ");
    }
    return res + ")";
}

const size_t& ParameterStateNew::operator[](const size_t& idx) const
{
    if (idx >= mDimensions.size())
        Logger::error << "Out of bounds!" << std::endl;
    return mData[idx];
}

size_t& ParameterStateNew::operator[](const size_t& idx)
{
    if (idx >= mData.size())
        Logger::error << "Out of bounds!" << std::endl;
    return mData[idx];
}

bool ParameterStateNew::operator==(const ParameterStateNew& b) const
{
    bool equal = true;
    for (int i = 0; i < mData.size(); i++)
    {
        equal &= b.mData[i] == mData[i];
    }
    return equal;
}

size_t* ParameterStateNew::data() {
    return &mData[0];
}

void EmpiricalBSSRDF::get_parameters(const ParameterStateNew& state, float& theta_i, optix::float2& r,
                                            optix::float2& theta_s, float& albedo, float& g, float& eta)
{
    if (!state.is_valid())
        return;
    theta_i = mParameters[theta_i_index][state[theta_i_index]];
    r.x = mParameters[r_index][state[r_index]];
    theta_s.x = mParameters[theta_s_index][state[theta_s_index]];
    r.y = mParameters[r_index][state[r_index] + 1];
    theta_s.y = mParameters[theta_s_index][state[theta_s_index] + 1];
    albedo = mParameters[albedo_index][state[albedo_index]];
    g = mParameters[g_index][state[g_index]];
    eta = mParameters[eta_index][state[eta_index]];
}

bool EmpiricalBSSRDF::get_single_index(const float val, const size_t idx, size_t& idx_res)
{
    auto res = std::find_if(std::begin(mParameters[idx]), std::end(mParameters[idx]),
                            [val](float& f)-> bool { return val - f < 1e-6f; });
    if (res == std::end(mParameters[idx]))
        return false;
    idx_res = static_cast<size_t>(std::distance(std::begin(mParameters[idx]), res));
    return true;
}

bool EmpiricalBSSRDF::get_index(const float theta_i, const float r, const float theta_s, const float albedo,
                                       const float g, const float eta,  ParameterStateNew& state)
{
    bool success = true;
    success &= get_single_index(theta_i, theta_i_index, state[theta_i_index]);
    success &= get_single_index(r, r_index, state[r_index]);
    success &= get_single_index(theta_s, theta_s_index, state[theta_s_index]);
    success &= get_single_index(albedo, albedo_index, state[albedo_index]);
    success &= get_single_index(g, g_index, state[g_index]);
    success &= get_single_index(eta, eta_index, state[eta_index]);
    state.mDimensions = get_material_geometry_dimensions();
    return success;
}


bool EmpiricalBSSRDF::get_material_index(const float albedo, const float g, const float eta,
                                                std::vector<size_t>& state)
{
    bool success = true;
    state.resize(3);
    success &= get_single_index(albedo, albedo_index, state[albedo_index]);
    success &= get_single_index(g, g_index, state[g_index]);
    success &= get_single_index(eta, eta_index, state[eta_index]);
    return success;
}

ParameterStateNew ParameterStateNew::next(const ParameterStateNew& state)
{
    ParameterStateNew val = state;
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

bool ParameterStateNew::is_valid() const
{
    return *this != invalid_index;
}

const std::map<size_t, std::string>& EmpiricalBSSRDF::get_parameter_names(OutputShape::Type type)
{
    return type == OutputShape::HEMISPHERE ? parameter_names_hemi : parameter_names_plane;
}

void EmpiricalBSSRDF::recalculate_dimensions() {
    for (int i = 0; i < number_dimensions; i++)
        mDimensions[i] = mParameters.at(i).size();
    mDimensions[r_index] -= 1; // This span deltas!
    mDimensions[theta_s_index] -= 1; // This span deltas!
}

EmpiricalBSSRDF::EmpiricalBSSRDF() {
    set_hemisphere_size(160,40);
    reset_parameters();
}

void EmpiricalBSSRDF::reset_parameters() {
    size_t h1 = get_dimension_1();
    size_t h2 = get_dimension_2();
    mParameters = default_parameters;
    set_hemisphere_size(h1,h2);
    recalculate_dimensions();
}

void EmpiricalBSSRDF::set_hemisphere_size(size_t h1, size_t h2) {
    mDimensions[dim_1_index] = h1;
    mDimensions[dim_2_index] = h2;
    if(mShape == OutputShape::HEMISPHERE)
    {
        mParameters[dim_1_index] = get_parameter_phi(h1);
        mParameters[dim_2_index] = get_parameter_theta(h2);
    }
    else
    {
        mParameters[dim_1_index] = get_parameter_y(h1);
        mParameters[dim_2_index] = get_parameter_x(h2);
    }
}

size_t multiply_array(const std::vector<size_t> & d)
{
    size_t res = 1;
    for(const size_t & v : d)
        res *= v;
    return res;
}

size_t EmpiricalBSSRDF::get_material_combinations() const {
    return multiply_array(get_material_dimensions());
}

size_t EmpiricalBSSRDF::get_material_geometry_combinations() const {
    return multiply_array(get_material_geometry_dimensions());
}

size_t EmpiricalBSSRDF::get_size() const {
    return multiply_array(get_material_geometry_hemisphere_dimensions());
}

ParameterStateNew EmpiricalBSSRDF::begin_index() const {
    return ParameterStateNew({0,0,0,0,0,0}, get_material_geometry_dimensions());
}


bool ParameterStateNew::increment(size_t src, size_t size, size_t& dst)
{
    dst = (src + 1) % size;
    return ((src + 1) / size) >= 1;
}

size_t ParameterStateNew::flatten() {
    return flatten_index(mData, mDimensions);
}
