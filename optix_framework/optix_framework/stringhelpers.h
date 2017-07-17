#ifndef stringhelpers_h__
#define stringhelpers_h__

#include <string>
#include <optix_math.h>
#include <vector>

// A bunch of helper functions to convert common optix types to std::string and vice versa
inline void split(std::vector<std::string> &tokens, const std::string &text, char sep) {
	int start = 0, end = 0;
	while ((end = (int)text.find(sep, start)) != std::string::npos) {
		tokens.push_back(text.substr(start, end - start));
		start = end + 1;
	}
	tokens.push_back(text.substr(start));
}
#define space std::string(" ")


template<typename T>
inline std::string tostring(T p) {return std::to_string(p);}

// One dimensional
template<>
inline std::string tostring(bool p) {return std::string(p? "true" : "false");}
template<>
inline std::string tostring(std::string p) {return p;}

// Two dimensional
template<>
inline std::string tostring(optix::float2 p) {return std::to_string(p.x) + space + std::to_string(p.y);}
template<>
inline std::string tostring(optix::int2 p) {return std::to_string(p.x) + space + std::to_string(p.y);}
template<>
inline std::string tostring(optix::uint2 p) {return std::to_string(p.x) + space + std::to_string(p.y);}


//Three dimensional
template<>
inline std::string tostring(optix::float3 p) {return std::to_string(p.x) + space + std::to_string(p.y) + space +std::to_string(p.z);}
template<>
inline std::string tostring(optix::int3 p) {return std::to_string(p.x) + space + std::to_string(p.y) + space +std::to_string(p.z);}
template<>
inline std::string tostring(optix::uint3 p) {return std::to_string(p.x) + space + std::to_string(p.y) + space +std::to_string(p.z);}

//Four dimensional
template<>
inline std::string tostring(optix::float4 p) {return std::to_string(p.x) + space + std::to_string(p.y) + space +std::to_string(p.z) + space +std::to_string(p.w);}
template<>
inline std::string tostring(optix::int4 p) {return std::to_string(p.x) + space + std::to_string(p.y) + space +std::to_string(p.z)+ space +std::to_string(p.w);}
template<>
inline std::string tostring(optix::uint4 p) {return std::to_string(p.x) + space + std::to_string(p.y) + space +std::to_string(p.z)+ space +std::to_string(p.w);}



template<typename T>
inline T tovalue(std::string p) {return (T)p;} //unsafe
template<>
inline bool tovalue(std::string p) {return (p == "true") ? true : false;}
template<>
inline int tovalue(std::string p) {return std::stoi(p);}
template<>
inline float tovalue(std::string p) {return (float)std::stod(p);}

// Two  dims
template<>
inline optix::float2 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens,p,' ');
	return optix::make_float2((float)std::stod(tokens[0]), (float)std::stod(tokens[1]));
}
template<>
inline optix::int2 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens,p,' ');
	return optix::make_int2(std::stoi(tokens[0]), std::stoi(tokens[1]));
}
template<>
inline optix::uint2 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens,p,' ');
	return optix::make_uint2((unsigned int)std::stoi(tokens[0]), (unsigned int)std::stoi(tokens[1]));
}


// Three  dims
template<>
inline optix::float3 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens,p,' ');
	return optix::make_float3((float)std::stod(tokens[0]), (float)std::stod(tokens[1]), (float)std::stod(tokens[2]));
}
template<>
inline optix::int3 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens,p,' ');
	return optix::make_int3(std::stoi(tokens[0]), std::stoi(tokens[1]), std::stoi(tokens[2]));
}
template<>
inline optix::uint3 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens,p,' ');
	return optix::make_uint3((unsigned int)std::stoi(tokens[0]), (unsigned int)std::stoi(tokens[1]), (unsigned int)std::stoi(tokens[2]));
}

// Four  dims
template<>
inline optix::float4 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens,p,' ');
	return optix::make_float4((float)std::stod(tokens[0]), (float)std::stod(tokens[1]), (float)std::stod(tokens[2]), (float)std::stod(tokens[3]));
}
template<>
inline optix::int4 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens,p,' ');
	return optix::make_int4(std::stoi(tokens[0]), std::stoi(tokens[1]), std::stoi(tokens[2]), std::stoi(tokens[3]));
}
template<>
inline optix::uint4 tovalue(std::string p) {
	std::vector<std::string> tokens;
	split(tokens,p,' ');
	return optix::make_uint4((unsigned int)std::stoi(tokens[0]), (unsigned int)std::stoi(tokens[1]), (unsigned int)std::stoi(tokens[2]), (unsigned int)std::stoi(tokens[3]));
}

#endif // stringhelpers_h__
