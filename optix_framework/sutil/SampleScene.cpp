
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

#if defined(__APPLE__)
#  include <OpenGL/gl.h>
#else
#  include <GL/glew.h>
#  if defined(_WIN32)
#    include <GL/wglew.h>
#  endif
#  include <GL/gl.h>
#endif

#include <SampleScene.h>

#include <optixu/optixu_math_stream_namespace.h>
#include <optixu/optixu.h>

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm>


using namespace optix;

//-----------------------------------------------------------------------------
// 
// SampleScene class implementation 
//
//-----------------------------------------------------------------------------

SampleScene::SampleScene()
  : m_camera_changed( true ), m_use_vbo_buffer( true )
{
  m_context = Context::create();
}

SampleScene::InitialCameraData::InitialCameraData( const std::string &camstr)
{
  std::istringstream istr(camstr);
  istr >> eye >> lookat >> up >> vfov;
}

Buffer SampleScene::createOutputBuffer( RTformat format,
                                        unsigned int width,
                                        unsigned int height = 1)
{

  Buffer buffer;

  if ( m_use_vbo_buffer )
  {
    /*
      Allocate first the memory for the gl buffer, then attach it to OptiX.
    */
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    size_t element_size;
    m_context->checkError(rtuGetSizeForRTformat(format, &element_size));
    glBufferData(GL_ARRAY_BUFFER, element_size * width * height, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, vbo);
    buffer->setFormat(format);
	if (width == 1)
	{
		buffer->setSize(height);
	}
	else if (height == 1)
	{
		buffer->setSize(width);
	}
	else
	{
		buffer->setSize(width, height);
	}
  }
  else {
    buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, format, width, height);
  }

  return buffer;
}

void SampleScene::cleanUp()
{
  m_context->destroy();
  m_context = 0;
}

void
SampleScene::trace( const RayGenCameraData& camera_data, bool& display )
{
  trace(camera_data);
}

void SampleScene::resize(unsigned int width, unsigned int height)
{
  try {
    Buffer buffer = getOutputBuffer();
    buffer->setSize( width, height );

    if(m_use_vbo_buffer)
    {
      buffer->unregisterGLBuffer();
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer->getGLBOId());
      glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer->getElementSize() * width * height, 0, GL_STREAM_DRAW);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      buffer->registerGLBuffer();
    }

  } catch( Exception& e ){
	  // FIXME LOGGER
    //sutilReportError( e.getErrorString().c_str() );
    exit(2);
  }

  // Let the user resize any other buffers
  doResize( width, height );
}

