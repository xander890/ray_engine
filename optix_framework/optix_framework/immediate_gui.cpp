#include "immediate_gui.h"
#define IMGUI_DISABLE_TEST_WINDOWS
#include "imgui/imgui_impl_glfw_gl3.h"

#include<GLFW/glfw3.h>



ImmediateGUI::ImmediateGUI(GLFWwindow * window)
{
    if (window != nullptr)
        win = window;
    else
        win = glfwCreateWindow(500, 500, "gui", nullptr, nullptr);

    context_win = glfwGetCurrentContext();
    glfwMakeContextCurrent(win);
	ImGui_ImplGlfwGL3_Init(win, false);
	glfwSetCharCallback(win, ImGui_ImplGlfwGL3_CharCallback);
	glfwSetScrollCallback(win, ImGui_ImplGlfwGL3_ScrollCallback);
	glfwSetKeyCallback(win, ImGui_ImplGlfwGL3_KeyCallback);
	glfwSetCharCallback(win, ImGui_ImplGlfwGL3_CharCallback);

    glfwMakeContextCurrent(context_win);

    context_win = nullptr;
}

ImmediateGUI::~ImmediateGUI()
{
	ImGui_ImplGlfwGL3_Shutdown();
}

bool ImmediateGUI::keyPressed(int key, int action, int modifier)
{
	ImGui_ImplGlfwGL3_KeyCallback(win, key, 0, action, modifier);
	return false;
}



bool ImmediateGUI::mousePressed(int x, int y, int button, int action, int mods)
{
	ImGui_ImplGlfwGL3_MouseButtonCallback(win, button, action, mods);

	return ImGui::IsMouseHoveringAnyWindow();
}

bool ImmediateGUI::mouseMoving(int x, int y)
{
	ImGuiIO& io = ImGui::GetIO();
	io.MousePos = ImVec2((float)x, (float)y);
	
	return true;
}


void ImmediateGUI::start_window(const char * name, int x, int y, int w, int h)
{
	context_win = glfwGetCurrentContext();
	glfwMakeContextCurrent(win);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) ;
    ImGui_ImplGlfwGL3_NewFrame();
	ImGui::SetNextWindowPos(ImVec2((float)0, (float)0), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2((float)w, (float)h), ImGuiCond_FirstUseEver);
	ImGui::Begin(name, (bool*)&visible, ImGuiWindowFlags_ShowBorders);
}

void ImmediateGUI::end_window()
{
    glfwSetWindowSize(win, (int)ImGui::GetWindowWidth() + 10, (int)ImGui::GetWindowHeight() + 10);
    ImGui::End();
    ImGui::Render();
    glfwSwapBuffers(win);
    glfwPollEvents();
    if(glfwWindowShouldClose(win)) { exit(0); }
	glfwMakeContextCurrent(context_win);
	context_win = nullptr;
}

