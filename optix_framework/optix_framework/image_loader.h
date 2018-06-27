#pragma once

#include <optixu/optixpp_namespace.h>
#include <string>
#include <iosfwd>
#include <memory>

//-----------------------------------------------------------------------------
//
// Utility functions
//
//-----------------------------------------------------------------------------

// Creates a TextureSampler object for the given image file.  If filename is 
// empty or the image loader fails, a 1x1 texture is created with the provided 
// default texture color.
class Texture;
std::unique_ptr<Texture> loadTexture( optix::Context context,
                                            const std::string& filename,
                                            const optix::float4& default_color );

std::unique_ptr<Texture> createOneElementSampler(optix::Context context, const optix::float4& default_color);
