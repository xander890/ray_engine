#pragma once
#include <string>
#include <optixu/optixpp_namespace.h>
#include <sample_scene.h>

class Mouse;
class Camera;
struct GLFWwindow;

/*
 * Generic class to encapsulate a Sample Scene and that provides an interface to the window system (GLFW in this case).
 */
class GLFWDisplay
{
public:

  static void init( int& argc, char** argv );
  static void run( const std::string& title, SampleScene* scene);

  static void set_requires_display( const bool requires_display ) { mRequiresDisplay = requires_display; }
  static bool is_display_available() { return mRequiresDisplay; }
  static void set_use_SRGB(bool enabled) { mUsesRGB = enabled; }

private:
  // Do the actual rendering to the display
  static void display_frame();

  // Cleans mUp the rendering context and quits.  If there wasn't error cleaning mUp, the 
  // return code is passed out, otherwise 2 is used as the return code.
  static void quit(int return_code=0);

  // callbacks
  static void display();
  static void key_pressed(GLFWwindow * window, int key, int scancode, int action, int modifier);
  static void mouse_button(GLFWwindow * window, int button, int action, int modifiers);
  static void mouse_moving(GLFWwindow * window, double xd, double yd);
  static void resize(GLFWwindow * window, int width, int height);

  static Mouse*         mMouse;
  static SampleScene*   mScene;
  static GLFWwindow * mWindow;

  static bool           mDisplayedFrames;
  static bool           mIssRGBSupported;
  static bool           mUsesRGB;
  static bool           mInitialized;

  static bool           mRequiresDisplay;
  static std::string    mTitle;
  static int            mAvailableGPUs;
};
