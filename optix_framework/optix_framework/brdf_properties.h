#pragma once

#define IMPROVED_ENUM_NAME BRDFType
#define IMPROVED_ENUM_LIST	ENUMITEM_VALUE(LAMBERTIAN,0) \
                            ENUMITEM_VALUE(TORRANCE_SPARROW,1) \
							ENUMITEM_VALUE(MERL,2)
#include "improved_enum.def"


struct BRDFGeometry
{
    optix::float3 texcoord;
    optix::float3 wi;
    optix::float3 wo;
    optix::float3 n;
};