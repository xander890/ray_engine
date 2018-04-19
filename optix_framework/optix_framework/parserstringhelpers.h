#ifndef stringhelpers_h__
#define stringhelpers_h__

#include <string>
#include <vector>
#include <sstream>

// A bunch of helper functions to convert common optix types to std::string and vice versa
inline void split(std::vector<std::string> &tokens, const std::string &text, char sep) {
	int start = 0, end = 0;
	while ((end = (int)text.find(sep, start)) != std::string::npos) {
		tokens.push_back(text.substr(start, end - start));
		start = end + 1;
	}
	tokens.push_back(text.substr(start));
}
#define space_token std::string(" ")

template<typename T>
inline std::string stringize(std::initializer_list<T> nums)
{
	std::stringstream ss;
	for (auto& s : nums)
		ss << s << " ";
	return ss.str();
}

template<typename T>
inline std::string tostring(T p) {return std::to_string(p);}


template<typename T>
inline std::string stringize(T* nums, std::size_t size, size_t precision = 8)
{
	std::stringstream ss;
	ss << std::fixed;
	ss.precision(precision);
	for (int i = 0; i < size; i++)
		ss << tostring(nums[i]) << " ";
	return ss.str();
}

template<>
inline std::string tostring(std::vector<float> p) { return stringize(p.data(), p.size()); }

// One dimensional
template<>
inline std::string tostring(bool p) {return std::string(p? "true" : "false");}
template<>
inline std::string tostring(const std::string & p) {return std::string(p);}
template<>
inline std::string tostring(std::string p) { return p; }
template<>
inline std::string tostring(const char * p) { return std::string(p); }



template<typename T>
inline T tovalue(std::string p) {return static_cast<T>(p);} //unsafe

template<>
inline const char* tovalue(std::string p) { return p.c_str(); }
template<>
inline bool tovalue(std::string p) {return (p == "true") ? true : false;}
template<>
inline int tovalue(std::string p) {return std::stoi(p);}
template<>
inline float tovalue(std::string p) {return (float)std::stod(p);}

// Two  dims

#include <optix_math.h>
#include <optixu/optixu_matrix_namespace.h>
// Two dimensional
template<>
inline std::string tostring(optix::float2 p) { return stringize({ p.x, p.y }); }
template<>
inline std::string tostring(optix::int2 p) { return stringize({ p.x, p.y }); }
template<>
inline std::string tostring(optix::uint2 p) { return stringize({ p.x, p.y }); }


//Three dimensional
template<>
inline std::string tostring(optix::float3 p) { return stringize({p.x,p.y,p.z}); }
template<>
inline std::string tostring(optix::int3 p) { return stringize({ p.x, p.y, p.z }); }
template<>
inline std::string tostring(optix::uint3 p) { return stringize({ p.x, p.y, p.z }); }

//Four dimensional
template<>
inline std::string tostring(optix::float4 p) { return stringize({ p.x, p.y, p.z, p.w }); }
template<>
inline std::string tostring(optix::int4 p) { return stringize({ p.x, p.y, p.z, p.w }); }
template<>
inline std::string tostring(optix::uint4 p) { return stringize({ p.x, p.y, p.z, p.w }); }

// Matrices
template<>
inline std::string tostring(optix::Matrix4x4 p) { return stringize(p.getData(), 16); }

template<>
inline std::string tostring(optix::Matrix2x2 p) { return stringize(p.getData(), 4); }

template<>
inline std::string tostring(optix::Matrix3x3 p) { return stringize(p.getData(), 9); }


template<>
inline optix::float2 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens, p, ' ');
	return optix::make_float2((float)std::stod(tokens[0]), (float)std::stod(tokens[1]));
}
template<>
inline optix::int2 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens, p, ' ');
	return optix::make_int2(std::stoi(tokens[0]), std::stoi(tokens[1]));
}
template<>
inline optix::uint2 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens, p, ' ');
	return optix::make_uint2((unsigned int)std::stoi(tokens[0]), (unsigned int)std::stoi(tokens[1]));
}


// Three  dims
template<>
inline optix::float3 tovalue(std::string p) {
	std::vector<float> tokens;
	std::stringstream ss(p);
	float a;
	while (ss >> a) tokens.push_back(a);
	while (tokens.size() < 3) tokens.push_back(0.0f);
	return optix::make_float3(tokens[0], tokens[1], tokens[2]);
}
template<>
inline optix::int3 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens, p, ' ');
	return optix::make_int3(std::stoi(tokens[0]), std::stoi(tokens[1]), std::stoi(tokens[2]));
}
template<>
inline optix::uint3 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens, p, ' ');
	return optix::make_uint3((unsigned int)std::stoi(tokens[0]), (unsigned int)std::stoi(tokens[1]), (unsigned int)std::stoi(tokens[2]));
}

// Four  dims
template<>
inline optix::float4 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens, p, ' ');
	return optix::make_float4((float)std::stod(tokens[0]), (float)std::stod(tokens[1]), (float)std::stod(tokens[2]), (float)std::stod(tokens[3]));
}
template<>
inline optix::int4 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens, p, ' ');
	return optix::make_int4(std::stoi(tokens[0]), std::stoi(tokens[1]), std::stoi(tokens[2]), std::stoi(tokens[3]));
}
template<>
inline optix::uint4 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens, p, ' ');
	return optix::make_uint4((unsigned int)std::stoi(tokens[0]), (unsigned int)std::stoi(tokens[1]), (unsigned int)std::stoi(tokens[2]), (unsigned int)std::stoi(tokens[3]));
}

template<>
inline optix::Matrix2x2 tovalue(std::string p) {
	std::vector<float> tokens;
	std::stringstream ss(p);
	float a;
	while (ss >> a) tokens.push_back(a);
	while (tokens.size() < 4) tokens.push_back(0.0f);

	// It will be copied in the Matrix, so we unsafely use stack data 
	return optix::Matrix2x2(&tokens[0]);
}

template<>
inline optix::Matrix3x3 tovalue(std::string p) {
	std::vector<float> tokens;
	std::stringstream ss(p);
	float a;
	while (ss >> a) tokens.push_back(a);
	while (tokens.size() < 9) tokens.push_back(0.0f);

	// It will be copied in the Matrix, so we unsafely use stack data 
	return optix::Matrix3x3(&tokens[0]);
}


template<>
inline optix::Matrix4x4 tovalue(std::string p) {
	std::vector<float> tokens;
	std::stringstream ss(p);
	float a;
	while (ss >> a) tokens.push_back(a);
	while (tokens.size() < 16) tokens.push_back(0.0f);

	// It will be copied in the Matrix, so we unsafely use stack data 
	return optix::Matrix4x4(&tokens[0]);
}

template<>
inline std::vector<float> tovalue(std::string p) {
	std::vector<float> tokens;
	std::stringstream ss(p);
	float a;
	while (ss >> a) tokens.push_back(a);
	return tokens;
}

template<>
inline std::vector<int> tovalue(std::string p) {
	std::vector<int> tokens;
	std::stringstream ss(p);
	int a;
	while (ss >> a) tokens.push_back(a);
	return tokens;
}


template<>
inline std::vector<unsigned int> tovalue(std::string p) {
	std::vector<unsigned int> tokens;
	std::stringstream ss(p);
	unsigned int a;
	while (ss >> a) tokens.push_back(a);
	return tokens;
}



#endif // stringhelpers_h__
