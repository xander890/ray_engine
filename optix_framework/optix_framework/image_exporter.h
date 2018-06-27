#pragma once

#include <optixu/optixpp_namespace.h>
#include <string>
#include <iosfwd>
#include <memory>
#include "texture.h"

//-----------------------------------------------------------------------------
//
// Utility functions
//
//-----------------------------------------------------------------------------

bool exportTexture(const std::string& filename, const std::unique_ptr<Texture> & tex, const int past_frames = 0);
bool exportTexture(const std::string& filename, optix::Buffer buf, const int past_frames = 0);
