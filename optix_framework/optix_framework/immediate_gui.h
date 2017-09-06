#pragma once
#include "imgui\imgui.h"

namespace ImmediateGUIDraw = ImGui;

class ImmediateGUI
{
public:

	ImmediateGUI(const char* name, int window_width, int window_height);

	virtual ~ImmediateGUI();


	bool keyPressed(unsigned char key, int x, int y);
	// Use this to add additional keys. Some are already handled but
	// can be overridden.  Should return true if key was handled, false otherwise.
	bool mousePressed(int button, int state, int x, int y);
	bool mouseMoving(int x, int y);
	void setWindowSize(int x, int y);

	void draw() const;

	void toggleVisibility() { visible = !visible; }
	bool isVisible() const
	{
		return visible;
	}	

private:
	bool visible = true;
	int window_width, window_height;

};