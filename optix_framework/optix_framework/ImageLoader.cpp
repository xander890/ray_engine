
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

std::unique_ptr<Texture> createOneElementSampler(optix::Context context, const optix::float4& default_color)
{
    std::unique_ptr<Texture> tex = std::make_unique<Texture>(context);
	tex->set_size(1);
    float data[4] = {default_color.x, default_color.y, default_color.z, default_color.w};
	tex->set_data(data, 4*sizeof(float));
    return tex;
}

std::unique_ptr<Texture> loadTexture( optix::Context context,
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
      return createOneElementSampler(context, make_float4(default_color,1));
  }
    
}

