#pragma once
#include "optix_world.h"
/*
 * Type of available BRDFs.
 *		LAMBERTIAN - classic lambertian model, using diffuse color only.
 *		TORRANCE_SPARROW - new rays are sampled according to the Torrance-Sparrow model, using roughness parameter.
 *		GGX				 - new rays are sampled according to GGX distribution.
 *		MERL			 - MERL data are used to render the BRDF. No importance sampling is possible in this particular case.
 */
#define IMPROVED_ENUM_NAME BRDFType
#define IMPROVED_ENUM_LIST	ENUMITEM_VALUE(LAMBERTIAN,0) \
                            ENUMITEM_VALUE(TORRANCE_SPARROW,1) \
							ENUMITEM_VALUE(MERL,2) \
							ENUMITEM_VALUE(GGX,3)
#include "improved_enum.def"


/*
 * BRDFGeometry class. Represents the input to a BRDF evaluation (normal, outgoing and incoming direction). Texture coordinates are provided for convenience.
 */
struct BRDFGeometry
{
    optix::float2 texcoord;
    optix::float3 wi;
    optix::float3 wo;
    optix::float3 n;
};