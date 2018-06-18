
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

#include <image_loader.h>
#include <raw_loader.h>
#include <fstream>
#include <algorithm>
#include <exception>
#include "IL/il.h"
#include "IL/ilu.h"

//-----------------------------------------------------------------------------
//  
//  Utility functions 
//
//-----------------------------------------------------------------------------

std::unique_ptr<Texture> createOneElementSampler(optix::Context context, const optix::float4& default_color)
{
    std::unique_ptr<Texture> tex = std::make_unique<Texture>(context);
	tex->set_size(1);
	tex->set_format(4, Texture::Format::FLOAT);
	float* buffer_data = new float[4];
	buffer_data[0] = default_color.x;
	buffer_data[1] = default_color.y;
	buffer_data[2] = default_color.z;
	buffer_data[3] = 1.0f;
	tex->set_data(buffer_data, 4 * sizeof(float));
	delete[] buffer_data;
    return tex;
}

bool loadDevilTexture(std::unique_ptr<Texture> &tex, optix::Context context, const std::string& filename)
{
	ilInit();

	ILuint	imgId;
	ilGenImages(1, &imgId);
	ilBindImage(imgId);

	tex = std::make_unique<Texture>(context);

	if(!ilLoadImage(filename.c_str()))
	{
		return false;
	}

	ILinfo ImageInfo;
	iluGetImageInfo(&ImageInfo);
	if (ImageInfo.Origin == IL_ORIGIN_UPPER_LEFT)
	{
		iluFlipImage();
	}

	printf("Width: %d  Height: %d  Depth: %d  Bpp: %d\n",
			ilGetInteger(IL_IMAGE_WIDTH),
			ilGetInteger(IL_IMAGE_HEIGHT),
			ilGetInteger(IL_IMAGE_DEPTH),
			ilGetInteger(IL_IMAGE_BITS_PER_PIXEL));

	int w = ilGetInteger(IL_IMAGE_WIDTH);
	int h = ilGetInteger(IL_IMAGE_HEIGHT);
	tex->set_size(w,h);
	tex->set_format(4, Texture::Format::FLOAT);

	float* data = new float[4*w*h];
	ilCopyPixels(0,0,0,w,h,1, IL_RGBA, IL_FLOAT, &data[0]);

	tex->set_data(data, 4*w*h*sizeof(float));

	ilDeleteImages(1, &imgId);
	ilShutDown();
	return true;
}


std::unique_ptr<Texture> loadTexture( optix::Context context, const std::string& filename, const optix::float4& default_color )
{
	std::string filename_lc = filename;
	std::transform(filename.begin(), filename.end(), filename_lc.begin(), ::tolower);
	size_t len = filename.length();
	bool success = false;
	std::unique_ptr<Texture> tex = nullptr;

	if(len >= 4)
	{
		std::string ext = filename_lc.substr(filename_lc.length() - 4);
		if (ext.compare(".raw") == 0)
		{
		    success = loadRAWTexture(tex, context, filename);
		}
		else
		{
		    success = loadDevilTexture(tex, context, filename);
		}
	}

	if(!success)
	{
		tex = createOneElementSampler(context, default_color);
	}

	return tex;

}

