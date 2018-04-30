
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

#include <PPMLoader.h>
#include <optixu/optixu_math_namespace.h>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace optix;

//-----------------------------------------------------------------------------
//  
//  PPMLoader class definition
//
//-----------------------------------------------------------------------------

PPMLoader::PPMLoader( const std::string& filename, const bool vflip )
  : m_nx( 0u ), m_ny( 0u ), m_max_val( 0u ), m_raster( 0 ), m_is_ascii(false)
{
  if ( filename.empty() ) return;
  
  size_t pos;
  std::string extension;
  if( (pos = filename.find_last_of( '.' )) != std::string::npos )
    extension = filename.substr( pos );
  if( extension != ".ppm" ) {
    std::cout << "PPMLoader( '" << filename << "' ) non-ppm file extension given '" << extension << "'" << std::endl;
    return;
  }

  // Open file
  try {
    std::ifstream file_in( filename.c_str(), std::ifstream::in | std::ifstream::binary );
    if ( !file_in ) {
		std::cout << "PPMLoader( '" << filename << "' ) failed to open file."
                << std::endl;
      return;
    }

    // Check magic number to make sure we have an ascii or binary PPM
    std::string line, magic_number;
    getLine( file_in, line );
    std::istringstream iss1( line );
    iss1 >> magic_number;
    if ( magic_number != "P6" && magic_number != "P3") {
		std::cout << "PPMLoader( '" << filename << "' ) unknown magic number: "
                << magic_number << ".  Only P3 and P6 supported." << std::endl;
      return;
    }
    if ( magic_number == "P3" ) {
      m_is_ascii = true;
    }

    // width, height
    getLine( file_in, line );
    std::istringstream iss2( line );
    iss2 >> m_nx >> m_ny;

    // max channel value
    getLine( file_in, line );
    std::istringstream iss3( line );
    iss3 >> m_max_val;

    m_raster = new(std::nothrow)  unsigned char[ m_nx*m_ny*3 ];
    if(m_is_ascii) {
      unsigned int num_elements = m_nx*m_ny*3;
      unsigned int count = 0;

      while(count < num_elements) {
        getLine( file_in, line );
        std::istringstream iss(line);

        while(iss.good()) {
          unsigned int c;
          iss >> c;
          m_raster[count++] = static_cast<unsigned char>(c);
        }
      }

    } else {
      file_in.read( reinterpret_cast<char*>( m_raster ), m_nx*m_ny*3 );
    }

    if(vflip) {
      unsigned char *m_raster2 = new(std::nothrow)  unsigned char[ m_nx*m_ny*3 ];
      for(unsigned int y2=m_ny-1, y=0; y<m_ny; y2--, y++) {
        for(unsigned int x=0; x<m_nx*3; x++)
          m_raster2[y2*m_nx*3+x] = m_raster[y*m_nx*3+x];
      }

      delete [] m_raster;
      m_raster = m_raster2;
    }
  } catch ( ... ) {
    std::cout << "PPMLoader( '" << filename << "' ) failed to load" << std::endl;
    m_raster = 0;
  }
}


PPMLoader::~PPMLoader()
{
  if ( m_raster ) delete[] m_raster;
}


bool PPMLoader::failed() const
{
  return m_raster == 0;
}


unsigned int PPMLoader::width() const
{
  return m_nx;
}


unsigned int PPMLoader::height() const
{
  return m_ny;
}


unsigned char* PPMLoader::raster() const
{
  return m_raster;
}


void PPMLoader::getLine( std::ifstream& file_in, std::string& s )
{
  for (;;) {
    if ( !std::getline( file_in, s ) )
      return;
    std::string::size_type index = s.find_first_not_of( "\n\r\t " );
    if ( index != std::string::npos && s[index] != '#' )
      break;
  }
}

  
//-----------------------------------------------------------------------------
//  
//  Utility functions 
//
//-----------------------------------------------------------------------------

std::unique_ptr<Texture> PPMLoader::loadTexture( optix::Context context,
                                              const float3& default_color,
                                              bool linearize_gamma)
{
  // lookup table for sRGB gamma linearization
  static unsigned char srgb2linear[256];
  // filling in a static lookup table for sRGB gamma linearization, standard formula for sRGB
  static bool srgb2linear_init = false;
  if (!srgb2linear_init) {
    srgb2linear_init = true;
    for (int i = 0; i < 256; i++) {
      float cs = i / 255.0f;
      if (cs <= 0.04045f)
        srgb2linear[i] = (unsigned char)(255.0f * cs / 12.92f + 0.5f);
      else
        srgb2linear[i] = (unsigned char)(255.0f * powf((cs + 0.055f)/1.055f, 2.4f) + 0.5f);
    }
  }

  // Create tex sampler and populate with default values

   std::unique_ptr<Texture> tex = std::make_unique<Texture>(context);

  if (failed() ) {

    float* buffer_data = new float[4];
    buffer_data[0] = default_color.x;
    buffer_data[1] = default_color.y;
    buffer_data[2] = default_color.z;
    buffer_data[3] = 1.0f;
    tex->set_data(buffer_data, 4 * sizeof(float));
    delete[] buffer_data;

    return tex;
  }

  const unsigned int nx = width();
  const unsigned int ny = height();

  // Create buffer and populate with PPM data
  tex->set_size(nx,ny);

  // Create buffer and populate with RAW data
  float* buffer_data = new float[nx*ny*4];

  float avg = 0.0f;

  for ( unsigned int i = 0; i < nx; ++i ) {
    for ( unsigned int j = 0; j < ny; ++j ) {

      unsigned int ppm_index = ( (ny-j-1)*nx + i )*3;
      unsigned int buf_index = ( (j     )*nx + i )*4;

      buffer_data[ buf_index + 0 ] = (raster()[ ppm_index + 0 ]) / 255.0f;
	  buffer_data[buf_index + 1] = (raster()[ppm_index + 1]) / 255.0f;
	  buffer_data[buf_index + 2] = (raster()[ppm_index + 2]) / 255.0f;
/*      if (linearize_gamma) {
        buffer_data[ buf_index + 0 ] = srgb2linear[buffer_data[ buf_index + 0 ]];
        buffer_data[ buf_index + 1 ] = srgb2linear[buffer_data[ buf_index + 1 ]];
        buffer_data[ buf_index + 2 ] = srgb2linear[buffer_data[ buf_index + 2 ]];
      }*/
      buffer_data[ buf_index + 3 ] = 1.0f;
	  avg += buffer_data[buf_index];
    }
  }

  tex->set_data(buffer_data, nx*ny*4*sizeof(float));
  delete[] buffer_data;


  return tex;
}

  
//-----------------------------------------------------------------------------
//  
//  Utility functions 
//
//-----------------------------------------------------------------------------

std::unique_ptr<Texture> loadPPMTexture( optix::Context context,
                                      const std::string& filename,
                                      const optix::float3& default_color )
{
  PPMLoader ppm( filename );
  return ppm.loadTexture(context, default_color );
}
