#include "immediate_gui.h"

#include "imgui/imgui_impl_glut.h"

#include "GL\glew.h"
#ifdef NOMINMAX
#undef NOMINMAX
#endif
#include "GL\freeglut.h"

ImmediateGUI::ImmediateGUI(const char * name, int window_width, int window_height) : window_width(window_width), window_height(window_height)
{
	ImGui_ImplGLUT_Init();
}

ImmediateGUI::~ImmediateGUI()
{
	ImGui_ImplGLUT_Shutdown();
}

bool ImmediateGUI::keyPressed(unsigned char key, int x, int y)
{
	ImGuiIO& io = ImGui::GetIO();
	io.AddInputCharacter(key);
	return false;
}

bool ImmediateGUI::mousePressed(int button, int state, int x, int y)
{
	ImGuiIO& io = ImGui::GetIO();
	io.MousePos = ImVec2((float)x, (float)y);

	if (state == GLUT_DOWN && (button == GLUT_LEFT_BUTTON))
		io.MouseDown[0] = true;
	else
		io.MouseDown[0] = false;

	if (state == GLUT_DOWN && (button == GLUT_RIGHT_BUTTON))
		io.MouseDown[1] = true;
	else
		io.MouseDown[1] = false;

	return ImGui::IsMouseHoveringAnyWindow();
}

bool ImmediateGUI::mouseMoving(int x, int y)
{
	ImGuiIO& io = ImGui::GetIO();
	io.MousePos = ImVec2((float)x, (float)y);
	
	return ImGui::IsMouseHoveringAnyWindow();
}

void ImmediateGUI::setWindowSize(int x, int y)
{
	window_width = x;
	window_height = y;
}

void ImmediateGUI::draw() const
{
	static bool show_test_window = false;
	static bool show_another_window = false;
	ImGui_ImplGLUT_NewFrame(window_width, window_height);

	// 1. Show a simple window
	// Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets appears in a window automatically called "Debug"
	{
		static float f = 0.0f;
		ImGui::Text("Hello, world!");
		ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
		if (ImGui::Button("Test Window")) show_test_window ^= 1;
		if (ImGui::Button("Another Window")) show_another_window ^= 1;
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	}

	// 2. Show another simple window, this time using an explicit Begin/End pair
	if (show_another_window)
	{
		ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
		ImGui::Begin("Another Window", &show_another_window);
		ImGui::Text("Hello");
		ImGui::End();
	}

	// 3. Show the ImGui test window. Most of the sample code is in ImGui::ShowTestWindow()
	if (show_test_window)
	{
		ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
		ImGui::ShowTestWindow(&show_test_window);
	}

	ImGui::Render();
}
