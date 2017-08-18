#pragma once
#include "host_device_common.h"
enum class CameraType { STANDARD_RT = 0, TONE_MAPPING = 1, COUNT = 2 };

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


//#define OVERRIDE_TRANSLUCENT_WITH_APPLE_JUICE
