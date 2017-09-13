#include "immediate_gui.h"
#define IMGUI_DISABLE_TEST_WINDOWS
#include "imgui/imgui_impl_glfw_gl3.h"

#include<glfw\glfw3.h>


ImmediateGUI::ImmediateGUI(GLFWwindow * window, const char * name)
{
	this->name = name;
	ImGui_ImplGlfwGL3_Init(window, false);
}

ImmediateGUI::~ImmediateGUI()
{
	ImGui_ImplGlfwGL3_Shutdown();
}

bool ImmediateGUI::keyPressed(unsigned char key, int x, int y)
{
	ImGuiIO& io = ImGui::GetIO();
	io.AddInputCharacter(key);
	return false;
}



bool ImmediateGUI::mousePressed(int x, int y, int button, int action, int mods)
{
	ImGuiIO& io = ImGui::GetIO();
	io.MousePos = ImVec2((float)x, (float)y);

	if (action == GLFW_PRESS && (button == GLFW_MOUSE_BUTTON_LEFT))
		io.MouseDown[0] = true;
	else
		io.MouseDown[0] = false;

	if (action == GLFW_PRESS && (button == GLFW_MOUSE_BUTTON_RIGHT))
		io.MouseDown[1] = true;
	else
		io.MouseDown[1] = false;

	return ImGui::IsMouseHoveringAnyWindow();
}

bool ImmediateGUI::mouseMoving(int x, int y)
{
	ImGuiIO& io = ImGui::GetIO();
	io.MousePos = ImVec2((float)x, (float)y);
	
	return true;
}

void ImmediateGUI::start_draw() const
{
	static bool show_test_window = false;
	static bool show_another_window = false;
	ImGui_ImplGlfwGL3_NewFrame();

	ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiSetCond_Always);
	ImGui::SetNextWindowSize(ImVec2(400, 600), ImGuiSetCond_Always);
	ImGui::Begin(name.c_str(), (bool*)&visible);
}

void ImmediateGUI::end_draw() const
{
	ImGui::End();
	ImGui::Render();
}
