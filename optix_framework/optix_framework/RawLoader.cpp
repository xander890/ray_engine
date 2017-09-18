
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

#include "RAWLoader.h"

#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>

//-----------------------------------------------------------------------------
//  
//  RAWLoader class definition
//
//-----------------------------------------------------------------------------


namespace {

	// The error class to throw
	struct RAWError {
		std::string Er;
		RAWError(const std::string &st = "RAWLoader error") : Er(st) {}
	};
};

RAWLoader::RAWLoader(const std::string& filename)
	: m_nx(0u), m_ny(0u), m_raster(0)
{
	if (filename.empty()) return;

	// Open file
	try {
		std::ifstream inf(filename.c_str(), std::ios::binary);

		if (!inf.is_open()) throw RAWError("Couldn't open file " + filename);

		std::string magic, comment;
		float exposure = 1.0f;

		unsigned int frame_number;

		std::string txt_file = filename.substr(0, filename.length() - 4) + ".txt";
		std::ifstream ifs_data(txt_file);
		if (ifs_data.bad())
			throw RAWError("Couldn't find txt file " + txt_file);
		ifs_data >> frame_number >> m_nx >> m_ny;

		float multipliers[3] = { 1, 1, 1 };
		for (int i = 0; i < 3; i++)
			if (!ifs_data.eof())
				ifs_data >> multipliers[i];
		ifs_data.close();


		// import image
		int size_image = m_ny * m_nx * 3;
		float* image = new float[size_image];
		std::ifstream ifs_image(filename, std::ifstream::binary);
		if (ifs_image.bad())
			throw RAWError("Couldn't open raw file " + filename);
		ifs_image.read(reinterpret_cast<char*>(image), size_image * sizeof(float));
		ifs_image.close();


		m_raster = new float[m_nx * m_ny * 4];

		for (unsigned int i = 0; i < m_nx * m_ny; ++i)
		{
			for (unsigned int j = 0; j < 3; ++j)
			{
				m_raster[i * 4 + j] = image[i * 3 + j] * multipliers[j];
			}
			m_raster[i * 4 + 3] = 1.0f;
		}
		delete[] image;
	}
	catch (const RAWError& err) {
		std::cerr << "RAWLoader( '" << filename << "' ) failed to load file: " << err.Er << '\n';
		delete[] m_raster;
		m_raster = 0;
	}
}


RAWLoader::~RAWLoader()
{
	delete[] m_raster;
}


bool RAWLoader::failed()const
{
	return m_raster == 0;
}


unsigned int RAWLoader::width()const
{
	return m_nx;
}


unsigned int RAWLoader::height()const
{
	return m_ny;
}


float* RAWLoader::raster()const
{
	return m_raster;
}


//-----------------------------------------------------------------------------
//  
//  Utility functions 
//
//-----------------------------------------------------------------------------

optix::TextureSampler loadRAWTexture(optix::Context& context,
	const std::string& filename,
	const optix::float3& default_color)
{
	// Create tex sampler and populate with default values
	optix::TextureSampler sampler = context->createTextureSampler();
	sampler->setWrapMode(0, RT_WRAP_REPEAT);
	sampler->setWrapMode(1, RT_WRAP_REPEAT);
	sampler->setWrapMode(2, RT_WRAP_REPEAT);
	sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	sampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
	sampler->setMaxAnisotropy(1.0f);
	sampler->setMipLevelCount(1u);
	sampler->setArraySize(1u);

	// Read in RAW, set texture buffer to empty buffer if fails
	RAWLoader RAW(filename);
	if (RAW.failed()) {

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

	const unsigned int nx = RAW.width();
	const unsigned int ny = RAW.height();

	// Create buffer and populate with RAW data
	optix::Buffer buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nx, ny);
	float* buffer_data = static_cast<float*>(buffer->map());

	float total = 0.0f;
	for (unsigned int i = 0; i < nx; ++i) {
		for (unsigned int j = 0; j < ny; ++j) {
			unsigned int buf_index = ((j)*nx + i) * 4;
			unsigned int RAW_index = ((j)*nx + i) * 4;

			buffer_data[buf_index + 0] = RAW.raster()[RAW_index + 0];
			buffer_data[buf_index + 1] = RAW.raster()[RAW_index + 1];
			buffer_data[buf_index + 2] = RAW.raster()[RAW_index + 2];
			buffer_data[buf_index + 3] = RAW.raster()[RAW_index + 3];
			total += (RAW.raster()[RAW_index + 0] + RAW.raster()[RAW_index + 1] + RAW.raster()[RAW_index + 2]) / 3.0f;
		}
	}
	std::cout << (total / nx) / ny << std::endl;

	buffer->unmap();

	sampler->setBuffer(0u, 0u, buffer);
	sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

	return sampler;
}
