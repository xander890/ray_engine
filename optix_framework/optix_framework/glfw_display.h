
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

#pragma once

#include <string>
#include <optixu/optixpp_namespace.h>
#include <sample_scene.h>

class Mouse;
class Camera;
struct GLFWwindow;

//-----------------------------------------------------------------------------
// 
// GLFWDisplay 
//
//-----------------------------------------------------------------------------



class GLFWDisplay
{
public:

  static void init( int& argc, char** argv );
  static void run( const std::string& title, SampleScene* scene);
  static void printUsage();

  static void setRequiresDisplay( const bool requires_display ) { m_requires_display = requires_display; }
  static bool isDisplayAvailable() { return m_requires_display; }
  static void setUseSRGB(bool enabled) { m_use_sRGB = enabled; }

private:
  // Do the actual rendering to the display
  static void displayFrame();

  // Cleans mUp the rendering context and quits.  If there wasn't error cleaning mUp, the 
  // return code is passed out, otherwise 2 is used as the return code.
  static void quit(int return_code=0);

  // callbacks
  static void display();
  static void keyPressed(GLFWwindow * window, int key, int scancode, int action, int modifier);
  static void mouseButton(GLFWwindow * window, int button, int section, int modifiers);
  static void mouseMotion(GLFWwindow * window, double x, double y);
  static void resize(GLFWwindow * window, int width, int height);

  static Mouse*         m_mouse;
  static SampleScene*   m_scene;
  static GLFWwindow * m_window;

  static bool           m_display_frames;
  static bool           m_sRGB_supported;
  static bool           m_use_sRGB;
  static bool           m_initialized;

  static bool           m_requires_display;
  static std::string    m_title;
  static int            m_num_devices;
};
