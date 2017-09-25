#pragma once
#include "host_device_common.h"
#include <algorithm>
#include <map>
enum class CameraType { STANDARD_RT = 0, TONE_MAPPING = 1, DEBUG = 2, COUNT = 3 };

template <typename E>
auto as_integer(E const value)
-> typename std::underlying_type<E>::type
{
	return static_cast<typename std::underlying_type<E>::type>(value);
}

#define IMPROVED_ENUM_NAME Scene
#define IMPROVED_ENUM_LIST ENUMITEM(OPTIX_ONLY) ENUMITEM(OPENGL_ONLY) ENUMITEM(HYBRID_SSBO) ENUMITEM(HYBRID_FIRST_PASS) 
#include "improved_enum.h"

#define IMPROVED_ENUM_NAME LightTypes
#define IMPROVED_ENUM_LIST ENUMITEM_VALUE(DIRECTIONAL_LIGHT, LIGHT_TYPE_DIR) ENUMITEM_VALUE(POINT_LIGHT, LIGHT_TYPE_POINT) ENUMITEM_VALUE(SKY_LIGHT, LIGHT_TYPE_SKY) ENUMITEM_VALUE(AREA_LIGHT, LIGHT_TYPE_AREA)
#include "improved_enum.h"

#define IMPROVED_ENUM_NAME BackgroundType
#define IMPROVED_ENUM_LIST ENUMITEM(CONSTANT_BACKGROUND) ENUMITEM(ENVIRONMENT_MAP) ENUMITEM(SKY_MODEL) 
#include "improved_enum.h"

#define IMPROVED_ENUM_NAME RenderingMethodType
#define IMPROVED_ENUM_LIST ENUMITEM(RECURSIVE_RAY_TRACING) ENUMITEM(AMBIENT_OCCLUSION) ENUMITEM(PATH_TRACING)
#include "improved_enum.h"

#define IMPROVED_ENUM_NAME PinholeCameraDefinitionType
#define IMPROVED_ENUM_LIST ENUMITEM(INVERSE_CAMERA_MATRIX) ENUMITEM(EYE_LOOKAT_UP_VECTORS) 
#include "improved_enum.h"

namespace SamplingMfpType
{
	enum SamplingMfpTypeEnum {
		X = 0, Y = 1, Z = 2, MIN = 3, MAX = 4, MEAN = 5, COUNT = 6
	};

	static std::map< SamplingMfpTypeEnum, std::string > info
	{
		{ X, "X" },{ Y, "Y" },{ Z, "Z" },{ MIN, "MIN" },{ MAX, "MAX" },{ MEAN, "MEAN" }
	};

	static __host__ std::string to_string(SamplingMfpTypeEnum e)
	{
		return info[e];
	}

	static __host__ std::string get_enum_string()
	{
		std::string r;
		for (int i = 0; i < SamplingMfpTypeEnum::COUNT; i++)
		{
			r += to_string(static_cast<SamplingMfpTypeEnum>(i)) + " ";
		}
		return r;
	}

	inline __host__ SamplingMfpTypeEnum to_enum(std::string e)
	{
		auto it = std::find_if(info.begin(), info.end(),
			[e](const std::pair<SamplingMfpTypeEnum, std::string> & t) -> bool { return e.compare(t.second) == 0; });
		if (it != info.end())
			return it->first;
		return COUNT;
	}
};



//#define OVERRIDE_TRANSLUCENT_WITH_APPLE_JUICE
