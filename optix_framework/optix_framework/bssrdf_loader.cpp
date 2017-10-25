#include "bssrdf_loader.h"
#include <fstream>
#include <sstream>
#include "logger.h"
#include "parserstringhelpers.h"
#include <algorithm>
#include <assert.h>
#include <string_utils.h>

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
		ss << parameter_delimiter << " " << names.at(p.first) << " " << tostring(p.second) << std::endl;
	}
	result = ss.str();
}

void string_to_parameters(const std::string & parse, const std::map<size_t, std::string> & names, std::map<size_t, std::vector<float>> & parameters)
{
	std::stringstream ss(parse);
	std::string str;
	while (std::getline(ss, str)) {
		if (str.size() >= parameter_delimiter.size() && str.substr(0, parameter_delimiter.size()) == parameter_delimiter)
		{
			std::stringstream ss(str.substr(parameter_delimiter.size()));
			std::string name;
			ss >> name;
			std::string s = ss.str();
			s.erase(0, name.length() + 1);
			auto res = std::find_if(std::begin(names), std::end(names), [&](const auto &pair)
			{
				return pair.second == name;
			});
			if (res != names.end())
				parameters[res->first] = tovalue<std::vector<float>>(s);
		}
	}
}

BSSRDFLoader::BSSRDFLoader(const std::string & filename, const std::map<size_t, std::string> & names)
{
	mFileName = filename;	

	if(!parse_header(names))
	{
		Logger::error << "BSSRDF header parsing failed." << std::endl;
	}
}

void BSSRDFLoader::get_dimensions(std::vector<size_t>& dimensions)
{
	dimensions.clear();
	dimensions.insert(dimensions.begin(), mDimensions.begin(), mDimensions.end());
}

size_t BSSRDFLoader::get_material_slice_size()
{
	return get_hemisphere_size() * mDimensions[albedo_index] * mDimensions[g_index] * mDimensions[eta_index];
}

size_t BSSRDFLoader::get_hemisphere_size()
{
	return mDimensions[phi_o_index] * mDimensions[theta_o_index];
}

const std::map<size_t, std::vector<float>>& BSSRDFLoader::get_parameters()
{
	return mParameters;
}

bool BSSRDFLoader::load_material_slice(float * bssrdf_data, const std::vector<size_t>& idx)
{
	if (idx.size() != 3)
	{
		return false;
		Logger::error << "Material index is 3 dimensional." << std::endl;
	}
#ifdef USE_SMALL_FILES
	size_t pos = 0;
	std::string filename = get_filename(mFileName, idx, mParameters);
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

bool BSSRDFLoader::load_hemisphere(float * bssrdf_data, const std::vector<size_t>& idx)
{
	if (idx.size() != 6)
	{
		Logger::error << "Hemisphere index is 6 dimensional." << std::endl;
		return false;
	}
#ifdef USE_SMALL_FILES
	std::vector<size_t> dims = mDimensions;
	dims[eta_index] = dims[albedo_index] = dims[g_index] = 1;
	size_t pos = flatten_index({ 0, 0, 0, idx[3], idx[4], idx[5], 0, 0 }, dims) * sizeof(float);
	std::string s = get_filename(mFileName, idx, mParameters);
	if (!file_exists(s))
	{
		//Logger::error << "File not found. " << s << std::endl;
		return false;
	}
	else
	{
		Logger::info << "File found. " << std::endl;
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
	
bool BSSRDFLoader::parse_header(const std::map<size_t, std::string> & names)
{

	std::ifstream file(mFileName, std::ofstream::in | std::ofstream::binary);

#define MAX_HEADER_SIZE 2048
	char header[MAX_HEADER_SIZE];
	file.read(header, MAX_HEADER_SIZE * sizeof(char));
	std::string str(header);

	size_t bssrdf_del = str.find(std::string("\n") + bssrdf_delimiter);

	bool parsed_dimensions = false;
	bool has_bssrdf_flag = false;

	mBSSRDFStart = str.find("\n", bssrdf_del + 1) + 1;

	
	string_to_parameters(str.substr(0, mBSSRDFStart), names, mParameters);

	std::stringstream ss(str);

	while (std::getline(ss, str)) {
		if (str.size() >= bssrdf_delimiter.size() && str.substr(0, bssrdf_delimiter.size()).compare(bssrdf_delimiter) == 0)
		{
			has_bssrdf_flag = true;
			break;
		}
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
		if (str.size() > 0 && str[0] == '#')
			continue;
	}
	return has_bssrdf_flag && parsed_dimensions;
}

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





BSSRDFExporter::BSSRDFExporter(const std::string & filename, const std::vector<size_t> & dimensions, const std::map<size_t, std::vector<float>> & parameters, const std::map<size_t, std::string> & names) : mFileName(filename), mDimensions(dimensions), mParameters(parameters), mNames(names)
{
	size_t total_size = sizeof(float);
	for (size_t element : dimensions)
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
	return get_hemisphere_size() * mDimensions[theta_i_index] * mDimensions[theta_s_index] * mDimensions[r_index];
}

size_t BSSRDFExporter::get_hemisphere_size()
{
	return mDimensions[phi_o_index] * mDimensions[theta_o_index];
}

void BSSRDFExporter::set_material_slice(const float * bssrdf_data, const std::vector<size_t>& idx)
{
	if (idx.size() != 3)
		Logger::error << "Material index is 3 dimensional." << std::endl;
#ifdef USE_SMALL_FILES
	size_t pos = 0;
	std::ofstream ofs;
	std::string filename = get_filename(mFileName, idx, mParameters);
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
	std::vector<size_t> dims = mDimensions;
	dims[eta_index] = dims[albedo_index] = dims[g_index] = 1;
	size_t pos = flatten_index({ 0, 0, 0, idx[3], idx[4], idx[5], 0, 0 }, dims) * sizeof(float);
	std::ofstream ofs;
	std::string filename = get_filename(mFileName, idx, mParameters);
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
	//ss.open(mFileName, std::ofstream::out);
	ss << "# BSSRDF file format (version 0.1)" << std::endl;
	ss << "# Index dimensions is at follows:" << std::endl;
	ss << size_delimiter << " ";
	for (int i = 0; i < 8; i++)
	{
		ss << mDimensions[i] << " ";
	}
	ss << std::endl;
	ss << "#eta\tg\talbedo\ttheta_s\tr\ttheta_i\tphi_o\ttheta_o" << std::endl;

	std::string params;
	parameters_to_string(mParameters, mNames, params);
	ss << params;

	ss << bssrdf_delimiter << std::endl;
	size_t t = ss.tellp();
	return ss.str();
}
