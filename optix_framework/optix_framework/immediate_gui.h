#pragma once
#include "imgui/imgui.h"
#include <string>
namespace ImmediateGUIDraw = ImGui;

struct GLFWwindow;

class ImmediateGUI
{
public:

	ImmediateGUI(GLFWwindow * window = nullptr);

	virtual ~ImmediateGUI();


	bool keyPressed(int key, int action, int modifier);
	// Use this to add additional keys. Some are already handled but
	// can be overridden.  Should return true if key was handled, false otherwise.
	bool mousePressed(int x, int y, int button, int action, int mods);
	bool mouseMoving(int x, int y);

	void start_window(const char * name, int x, int y, int w, int h);
	void end_window();

	void toggleVisibility() { visible = !visible; }
	bool isVisible() const
	{
		return visible;
	}	

private:
	bool visible = true;
	GLFWwindow * win;
	GLFWwindow * context_win = nullptr;
};
