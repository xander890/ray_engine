
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

#include <GL/glew.h>
#include <GL/wglew.h>
#ifdef NOMINMAX
#undef NOMINMAX
#endif
#include <GL/glut.h>

#define NOMINMAX
#include <GLUTDisplay.h>
#include <Mouse.h>


#include <optixu/optixu_math_stream_namespace.h>

#include <iostream>
#include <cstdio> //sprintf
#include <sstream>
#include "logger.h"

// #define NVTX_ENABLE enables the nvToolsExt stuff from Nsight in NsightHelper.h
//#define NVTX_ENABLE

using namespace optix;

//-----------------------------------------------------------------------------
// 
// GLUTDisplay class implementation 
//-----------------------------------------------------------------------------

Mouse*         GLUTDisplay::m_mouse = 0;
PinholeCamera* GLUTDisplay::m_camera = 0;
SampleScene*   GLUTDisplay::m_scene = 0;
bool           GLUTDisplay::m_display_frames = true;

unsigned int   GLUTDisplay::m_texId = 0;
bool           GLUTDisplay::m_sRGB_supported = false;
bool           GLUTDisplay::m_use_sRGB = false;

bool           GLUTDisplay::m_initialized = false;
std::string    GLUTDisplay::m_title = "";

bool            GLUTDisplay::m_requires_display = true;
bool            GLUTDisplay::m_benchmark_no_display = false;


void GLUTDisplay::printUsage()
{

}

void GLUTDisplay::init( int& argc, char** argv )
{
  m_initialized = true;

  if (!m_benchmark_no_display)
  {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  }
}

void GLUTDisplay::run( const std::string& title, SampleScene* scene, contDraw_E continuous_mode )
{
  if ( !m_initialized ) {
    std::cerr << "ERROR - GLUTDisplay::run() called before GLUTDisplay::init()" << std::endl;
    exit(2);
  }
  m_scene = scene;
  m_title = title;

  // Initialize GLUT and GLEW first. Now initScene can use OpenGL and GLEW.
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutInitWindowSize( 128, 128 );
  glutInitWindowPosition(100,100);
  glutCreateWindow( m_title.c_str() );
  glutHideWindow();
  glewInit();
  if (glewIsSupported( "GL_EXT_texture_sRGB GL_EXT_framebuffer_sRGB")) {
    m_sRGB_supported = true;
  }
  // Turn off vertical sync
  wglSwapIntervalEXT(0);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

  // If m_app_continuous_mode was already set to CDBenchmark* on the command line then preserve it.

  int buffer_width;
  int buffer_height;
  try {
    // Set up scene
    SampleScene::InitialCameraData camera_data;
    m_scene->initScene( camera_data );

    // Initialize camera according to scene params
    m_camera = new PinholeCamera( camera_data.eye,
                                 camera_data.lookat,
                                 camera_data.up,
                                 -1.0f, // hfov is ignored when using keep vertical
                                 camera_data.vfov,
                                 PinholeCamera::KeepVertical );

    Buffer buffer = m_scene->getOutputBuffer();
    RTsize buffer_width_rts, buffer_height_rts;
    buffer->getSize( buffer_width_rts, buffer_height_rts );
    buffer_width  = static_cast<int>(buffer_width_rts);
    buffer_height = static_cast<int>(buffer_height_rts);
    m_mouse = new Mouse( m_camera, buffer_width, buffer_height );
  } catch( Exception& e ){
    Logger::error << ( e.getErrorString().c_str() );
    exit(2);
  }

  // Initialize state
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1, 0, 1, -1, 1 );
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glViewport(0, 0, buffer_width, buffer_height);

  glutShowWindow();

  // reshape window to the correct window resize
  glutReshapeWindow( buffer_width, buffer_height);

  // Set callbacks
  glutKeyboardFunc(keyPressed);
  glutDisplayFunc(display);
  glutMouseFunc(mouseButton);
  glutMotionFunc(mouseMotion);
  glutReshapeFunc(resize);
  glutIdleFunc(idle);

  // Enter main loop
  glutMainLoop();
}

void GLUTDisplay::setCamera(SampleScene::InitialCameraData& camera_data)
{
  m_camera->setParameters(camera_data.eye,
                         camera_data.lookat,
                         camera_data.up,
                         camera_data.hfov, 
                         camera_data.vfov,
                         PinholeCamera::KeepVertical );
  glutPostRedisplay();  
}

void GLUTDisplay::postRedisplay()
{
  glutPostRedisplay();
}

void GLUTDisplay::keyPressed(unsigned char key, int x, int y)
{
  try {
    if( m_scene->keyPressed(key, x, y) ) {
      glutPostRedisplay();
      return;
    }
  } catch( Exception& e ){
    Logger::error << ( e.getErrorString().c_str() );
    exit(2);
  }

  switch (key) {
  case 27: // esc
  case 'q':
    quit();

  case 'c':
    float3 eye, lookat, up;
    float hfov, vfov;

    m_camera->getEyeLookUpFOV(eye, lookat, up, hfov, vfov);
    std::cerr << '"' << eye << lookat << up << vfov << '"' << std::endl;
    break;
  default:
    return;
  }
}


void GLUTDisplay::mouseButton(int button, int state, int x, int y)
{
	y += 9;
	if (!m_scene->mousePressed(button, state, x, y))
	{
		m_mouse->handleMouseFunc(button, state, x, y, glutGetModifiers());
		m_scene->signalCameraChanged();
	}
    glutPostRedisplay();
}


void GLUTDisplay::mouseMotion(int x, int y)
{
	y += 9;
	if (!m_scene->mouseMoving(x, y))
	{
		m_mouse->handleMoveFunc(x, y);
		m_scene->signalCameraChanged();		
	}
	glutPostRedisplay();
}


void GLUTDisplay::resize(int width, int height)
{
  // disallow size 0
  width  = max(1, width);
  height = max(1, height);

  m_scene->signalCameraChanged();
  m_mouse->handleResize( width, height );

  try {
    m_scene->resize(width, height);
  } catch( Exception& e ){
    Logger::error << ( e.getErrorString().c_str() );
    exit(2);
  }

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1, 0, 1, -1, 1);
  glViewport(0, 0, width, height);
  glutPostRedisplay();
}


void GLUTDisplay::idle()
{
  glutPostRedisplay();
}


void GLUTDisplay::displayFrame()
{
  GLboolean sRGB = GL_FALSE;
  if (m_use_sRGB && m_sRGB_supported) {
    glGetBooleanv( GL_FRAMEBUFFER_SRGB_CAPABLE_EXT, &sRGB );
    if (sRGB) {
      glEnable(GL_FRAMEBUFFER_SRGB_EXT);
    }
  }

  // Draw the resulting image
  Buffer buffer = m_scene->getOutputBuffer(); 
  RTsize buffer_width_rts, buffer_height_rts;
  buffer->getSize( buffer_width_rts, buffer_height_rts );
  int buffer_width  = static_cast<int>(buffer_width_rts);
  int buffer_height = static_cast<int>(buffer_height_rts);
  RTformat buffer_format = buffer->getFormat();


  unsigned int vboId = 0;
  if( m_scene->usesVBOBuffer() )
    vboId = buffer->getGLBOId();

  if (vboId)
  {

    if (!m_texId)
    {
      glGenTextures( 1, &m_texId );
      glBindTexture( GL_TEXTURE_2D, m_texId);

      // Change these to GL_LINEAR for super- or sub-sampling
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

      // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

      glBindTexture( GL_TEXTURE_2D, 0);
    }

    glBindTexture( GL_TEXTURE_2D, m_texId );

    // send pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vboId);

    RTsize elementSize = buffer->getElementSize();
    if      ((elementSize % 8) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if ((elementSize % 4) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if ((elementSize % 2) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else                             glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    {
      if(buffer_format == RT_FORMAT_UNSIGNED_BYTE4) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, buffer_width, buffer_height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
      } else if(buffer_format == RT_FORMAT_FLOAT4) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, buffer_width, buffer_height, 0, GL_RGBA, GL_FLOAT, 0);
      } else if(buffer_format == RT_FORMAT_FLOAT3) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, buffer_width, buffer_height, 0, GL_RGB, GL_FLOAT, 0);
      } else if(buffer_format == RT_FORMAT_FLOAT) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, buffer_width, buffer_height, 0, GL_LUMINANCE, GL_FLOAT, 0);
      } else {
        assert(0 && "Unknown buffer format");
      }
    }
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

    glEnable(GL_TEXTURE_2D);

    // Initialize offsets to pixel center sampling.

    float u = 0.5f/buffer_width;
    float v = 0.5f/buffer_height;

    glBegin(GL_QUADS);
    glTexCoord2f(u, v);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, v);
    glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f - u, 1.0f - v);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(u, 1.0f - v);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    glDisable(GL_TEXTURE_2D);
  } else {
    GLvoid* imageData = buffer->map();
    assert( imageData );

    GLenum gl_data_type = GL_FALSE;
    GLenum gl_format = GL_FALSE;

    switch (buffer_format) {
          case RT_FORMAT_UNSIGNED_BYTE4:
            gl_data_type = GL_UNSIGNED_BYTE;
            gl_format    = GL_BGRA;
            break;

          case RT_FORMAT_FLOAT:
            gl_data_type = GL_FLOAT;
            gl_format    = GL_LUMINANCE;
            break;

          case RT_FORMAT_FLOAT3:
            gl_data_type = GL_FLOAT;
            gl_format    = GL_RGB;
            break;

          case RT_FORMAT_FLOAT4:
            gl_data_type = GL_FLOAT;
            gl_format    = GL_RGBA;
            break;

          default:
            fprintf(stderr, "Unrecognized buffer data type or format.\n");
            exit(2);
            break;
    }

    RTsize elementSize = buffer->getElementSize();
    int align = 1;
    if      ((elementSize % 8) == 0) align = 8; 
    else if ((elementSize % 4) == 0) align = 4;
    else if ((elementSize % 2) == 0) align = 2;
    glPixelStorei(GL_UNPACK_ALIGNMENT, align);

    glDrawPixels( static_cast<GLsizei>( buffer_width ), static_cast<GLsizei>( buffer_height ),
      gl_format, gl_data_type, imageData);

    buffer->unmap();

  }
  if (m_use_sRGB && m_sRGB_supported && sRGB) {
    glDisable(GL_FRAMEBUFFER_SRGB_EXT);
  }
}

void GLUTDisplay::display()
{

  bool display_requested = true;

  try {
    // render the scene
    float3 eye, U, V, W;
    m_camera->getEyeUVW( eye, U, V, W );
    // Don't be tempted to just start filling in the values outside of a constructor, 
    // because if you add a parameter it's easy to forget to add it here.

    SampleScene::RayGenCameraData camera_data( eye, U, V, W );

    {
      m_scene->trace( camera_data, display_requested );
    }

    if( display_requested && m_display_frames ) {
      // Only enable for debugging
      // glClearColor(1.0, 0.0, 0.0, 0.0);
      // glClear(GL_COLOR_BUFFER_BIT);

      displayFrame();
    }
  } catch( Exception& e ){
    Logger::error << ( e.getErrorString().c_str() );
    exit(2);
  }
  m_scene->postDrawCallBack();

  std::string debug;

  if ( display_requested && m_display_frames ) {
    // Swap buffers
    glutSwapBuffers();
  }
}

void GLUTDisplay::quit(int return_code)
{
  try {
    if(m_scene)
    {
      m_scene->cleanUp();
      if (m_scene->getContext().get() != 0)
      {
        Logger::error << ( "Derived scene class failed to call SampleScene::cleanUp()" );
        exit(2);
      }
    }
    exit(return_code);
  } catch( Exception& e ) {
    Logger::error << ( e.getErrorString().c_str() );
    exit(2);
  }
}
