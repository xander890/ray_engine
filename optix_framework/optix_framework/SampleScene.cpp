
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
#include "logger.h"

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

void SampleScene::clean_up()
{
  m_context->destroy();
  m_context = 0;
}


void SampleScene::resize(unsigned int width, unsigned int height)
{
  try {
    Buffer buffer = get_output_buffer();
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
	Logger::error <<  e.getErrorString();
    exit(2);
  }

  // Let the user resize any other buffers
  do_resize( width, height );
}

