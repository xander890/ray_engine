
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <ImageLoader.h>
#include <PPMLoader.h>
#include <HDRLoader.h>
#include <RawLoader.h>
#include <fstream>
#include <algorithm>
#include <exception>


//-----------------------------------------------------------------------------
//  
//  Utility functions 
//
//-----------------------------------------------------------------------------

optix::TextureSampler createOneElementSampler(optix::Context context, const optix::float3& default_color)
{
    optix::TextureSampler sampler = context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_REPEAT);
    sampler->setWrapMode(1, RT_WRAP_REPEAT);
    sampler->setWrapMode(2, RT_WRAP_REPEAT);
    sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
    sampler->setMaxAnisotropy(1.0f);
    sampler->setMipLevelCount(1u);
    sampler->setArraySize(1u);

    // Create buffer with single texel set to default_color
    optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 1u, 1u);
    float* buffer_data = static_cast<float*>(buffer->map());
    buffer_data[0] = default_color.x;
    buffer_data[1] = default_color.y;
    buffer_data[2] = default_color.z;
    buffer_data[3] = 1.0f;
    buffer->unmap();

    sampler->setBuffer(0u, 0u, buffer);
    // Although it would be possible to use nearest filtering here, we chose linear
    // to be consistent with the textures that have been loaded from a file. This
    // allows OptiX to perform some optimizations.

    sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

    return sampler;
}

optix::TextureSampler loadTexture( optix::Context context,
                                            const std::string& filename,
                                            const optix::float3& default_color )
{
	std::string filename_lc = filename;
	std::transform(filename.begin(), filename.end(), filename_lc.begin(), ::tolower);
  size_t len = filename.length();

  if(len >= 4) {
	  std::string ext = filename_lc.substr(filename_lc.length() - 4);
	  if (ext.compare(".raw") == 0)
	  {
		  return loadRAWTexture(context, filename, default_color);
	  }
	  else if (ext.compare(".hdr") == 0)
	  {
		  return loadHDRTexture(context, filename, default_color);
	  }
	  else if (ext.compare(".ppm") == 0)
	  {
		  return loadPPMTexture(context, filename, default_color);
	  }
	  else
	  {
		  throw std::runtime_error("Unable to find a compatible format. Please use .ppm, .hdr or .raw.");
	  }

  }
  else
  {
      return createOneElementSampler(context, default_color);
  }
    
}

