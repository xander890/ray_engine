#include "bssrdf_loader.h"
#include <fstream>
#include <sstream>
#include "logger.h"
#include "parsing_utils.h"
#include <algorithm>
#include <assert.h>
#include <string_utils.h>
#include "empirical_bssrdf_common.h"

std::vector<float> convert_to_rad(const std::vector<float> & vec)
{
    std::vector<float> n;
    for(auto & v : vec)
    {
        n.push_back(deg2rad(v));
    }
    return n;
}

std::string get_filename(const std::string & filename, const std::vector<size_t> & idx, const std::map<size_t, std::vector<float>> & params)
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

BSSRDFImporter::BSSRDFImporter(const std::string & filename)
{
	mFileName = filename;

	if (!parse_header())
	{
		Logger::error << "BSSRDF header parsing failed." << std::endl;
	}
}

void BSSRDFImporter::get_dimensions(std::vector<size_t>& dimensions)
{
	dimensions.clear();
	dimensions.insert(dimensions.begin(), mDimensions.begin(), mDimensions.end());
}

size_t BSSRDFImporter::get_material_slice_size()
{
	return get_hemisphere_size() * mDimensions[theta_i_index] * mDimensions[theta_s_index] * mDimensions[r_index];
}

size_t BSSRDFImporter::get_hemisphere_size()
{
	return get_hemisphere_dimension_2()* get_hemisphere_dimension_1();
}

const std::map<size_t, std::vector<float>>& BSSRDFImporter::get_parameters()
{
	return mParameters;
}

bool BSSRDFImporter::load_material_slice(float * bssrdf_data, const std::vector<size_t>& idx)
{
	if (idx.size() != 3)
	{
        Logger::error << "Material index is 3 dimensional." << std::endl;
		return false;
	}
#ifdef USE_SMALL_FILES
	size_t pos = 0;
	std::string filename = get_filename(mFileName, idx, mParameters);
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

bool BSSRDFImporter::load_hemisphere(float * bssrdf_data, const std::vector<size_t>& idx)
{
	if (idx.size() != 6)
	{
		Logger::error << "Hemisphere index is 6 dimensional." << std::endl;
		return false;
	}
#ifdef USE_SMALL_FILES
	std::vector<size_t> dims = std::vector<size_t>(mDimensions.begin(), mDimensions.begin() + 6);
	dims[eta_index] = dims[albedo_index] = dims[g_index] = 1;
	size_t pos = flatten_index({ 0, 0, 0, idx[3], idx[4], idx[5] }, dims) * get_hemisphere_size() * sizeof(float);
	std::string s = get_filename(mFileName, idx, mParameters);
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
	
bool BSSRDFImporter::parse_header()
{
	std::ifstream file(mFileName);
    std::string str;

	bool parsed_dimensions = false;
	bool shape_found = false;

	while (std::getline(file, str)) {
		if (str.size() >= size_delimiter.size() && str.substr(0, size_delimiter.size()).compare(size_delimiter) == 0)
		{
			std::stringstream ss (str.substr(size_delimiter.size()));
			for (int i = 0; i < 8; i++)
			{
				size_t size;
				ss >> size; 
				if (ss.fail())
					return false;
				mDimensions.push_back(size);
			}
			parsed_dimensions = true;
		}
		if (str.size() >= shape_delimiter.size() && str.substr(0, shape_delimiter.size()).compare(shape_delimiter) == 0)
		{
			if(shape_found)
				Logger::warning << "Shape inconsistency found. Be careful to specify the shape before the parameters!" << std::endl;
			mOutputShape = OutputShape::to_enum(str.substr(shape_delimiter.size()));
			shape_found = true;
		}
        if (str.size() >= parameter_delimiter.size() && str.substr(0, parameter_delimiter.size()) == parameter_delimiter)
        {
			if(!shape_found)
			{
				Logger::error << "Shape has not been found. Assuming hemisphere!" << std::endl;
				mOutputShape = OutputShape::HEMISPHERE;
				shape_found = true;
			}
            std::stringstream ss(str.substr(parameter_delimiter.size()));
            std::string name;
            ss >> name;
            std::string s = ss.str();
            s.erase(0, name.length() + 1);
			auto names = BSSRDFParameterManager::get_parameter_names(mOutputShape);
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


BSSRDFExporter::BSSRDFExporter(const OutputShape::Type shape, const std::string &filename, const BSSRDFParameterManager &manager, size_t size_theta_o, size_t size_phi_o): mManager(manager), mFileName(filename)
{
	mOutputShape = shape;
	mDim2Size = size_theta_o;
	mDim1Size = size_phi_o;
	size_t total_size = sizeof(float);
	for (size_t element : mManager.get_dimensions())
		total_size *= element;

	mHeader = create_header();

#ifdef USE_SMALL_FILES
	std::ofstream of;
	of.open(filename, std::ofstream::out);
	of << mHeader;
	of.close();
#else
	std::ofstream of;
	of.open(mFileName, std::ofstream::out);
	of << mHeader;
	of.seekp(mHeader.size() + total_size - 1);
	of.put('\0');
	of.close();
#endif
}

size_t BSSRDFExporter::get_material_slice_size()
{
	auto dims = mManager.get_dimensions();
	return get_hemisphere_size() * dims[theta_i_index] * dims[theta_s_index] * dims[r_index];
}

size_t BSSRDFExporter::get_hemisphere_size()
{
	return mDim2Size * mDim1Size;
}

void BSSRDFExporter::set_material_slice(const float * bssrdf_data, const std::vector<size_t>& idx)
{
	if (idx.size() != 3)
		Logger::error << "Material index is 3 dimensional." << std::endl;
#ifdef USE_SMALL_FILES
	size_t pos = 0;
	std::ofstream ofs;
	std::string filename = get_filename(mFileName, idx, mManager.parameters);
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

void BSSRDFExporter::set_hemisphere(const float * bssrdf_data, const std::vector<size_t>& idx)
{
	if (idx.size() != 6)
		Logger::error << "Hemisphere index is 6 dimensional." << std::endl;
#ifdef USE_SMALL_FILES
	std::vector<size_t> dims = mManager.get_dimensions();
	dims[eta_index] = dims[albedo_index] = dims[g_index] = 1;
	size_t pos = flatten_index({ 0, 0, 0, idx[3], idx[4], idx[5]}, dims) * get_hemisphere_size() * sizeof(float);
	std::ofstream ofs;
	std::string filename = get_filename(mFileName, idx, mManager.parameters);
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

std::string BSSRDFExporter::create_header()
{
	std::stringstream ss;
	ss << "# BSSRDF file format (version 0.2)" << std::endl;
	ss << "# Index dimensions is at follows:" << std::endl;
	ss << size_delimiter << " ";
	auto di = mManager.get_dimensions();
	for (int i = 0; i < di.size(); i++)
	{
		ss << di[i] << " ";
	}
	ss << mDim1Size << " " << mDim2Size;
	ss << std::endl;

	auto names = BSSRDFParameterManager::get_parameter_names(mOutputShape);

	std::string dim_1_token = names[dim_1_index];
	std::string dim_2_token = names[dim_2_index];

	ss << "#eta g albedo theta_s r theta_i " << dim_1_token << " " << dim_2_token << std::endl;
	ss << shape_delimiter << " " << OutputShape::to_string(mOutputShape) << std::endl;

	std::string params;
	parameters_to_string(mManager.parameters, names, params);
	ss << params;

    ss << parameter_delimiter << " " << dim_1_token << "_index";
    for(int i = 0; i <= mDim1Size; i++)
	{
        float normalized = static_cast<float>(i) / mDim1Size;
		if(mOutputShape == OutputShape::HEMISPHERE)
		{
			float theta_o, phi_o;
			get_angles_polar(normalized, 0, phi_o, theta_o);
			ss << " " << std::to_string(phi_o);
		}
		else
		{
			float x = normalized * 2 - 1;
			ss << " " << std::to_string(x);
		}
	}
    ss << std::endl;

    ss << parameter_delimiter << " " << dim_2_token << "_index";
    for(int i = 0; i <= mDim2Size; i++)
    {
        float normalized = static_cast<float>(i) / mDim2Size;

		if(mOutputShape == OutputShape::HEMISPHERE)
		{
			float theta_o, phi_o;
			get_angles_polar(0, normalized, phi_o, theta_o);
			ss << " " << std::to_string(theta_o);
		}
		else
		{
			float y = normalized * 2 - 1;
			ss << " " << std::to_string(y);
		}
    }
    ss << std::endl;
    ss << bssrdf_delimiter << std::endl;
	return ss.str();
}

