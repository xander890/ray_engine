#include "gui.h"
#include <optixu/optixu_vector_functions.h>

#include "anttweak/AntTweakBar.h"
#include <GL/glut.h>
#include <iostream>
#include <sstream>


GUI::GUI(const char* name, int window_width, int window_height)
{
	TwInit(TW_OPENGL_CORE, nullptr);
	bar = TwNewBar(name);
	auto define = std::string(name) + " size='400 600' color='96 216 224' ";
	TwDefine(define.c_str()); // change default tweak bar size and color
	TwWindowSize(window_width, window_height);
	glutPassiveMotionFunc(reinterpret_cast<GLUTmousemotionfun>(TwEventMouseMotionGLUT));
	// - Directly redirect GLUT key events to AntTweakBar
	glutSpecialFunc(reinterpret_cast<GLUTspecialfun>(TwEventSpecialGLUT));
	
		
	TwGLUTModifiersFunc(glutGetModifiers);
	const char * error = TwGetLastError();
	if (error != nullptr)
		std::cout << "GUI error: " << error << std::endl;	
}

GUI::~GUI()
{
}

bool GUI::keyPressed(unsigned char key, int x, int y)
{
	return TwEventKeyboardGLUT(key, x, y);
}

bool GUI::mousePressed(int button, int state, int x, int y)
{
    if (isVisible())
    	return TwEventMouseButtonGLUT(button, state, x, y);
    return false;
}

bool GUI::mouseMoving(int x, int y)
{
    if (isVisible())
        return TwEventMouseMotionGLUT(x, y);
    return false;
}

void GUI::setWindowSize(int x, int y)
{
	TwWindowSize(x, y);
}

void GUI::addIntVariable(const char* name, int* var, const char* group, int min, int max, int step) const
{
	group =  strcmp(group, "") == 0? "main" : group;
	std::stringstream ss;
	ss << "group='" << group << "'" << " min=" << min << " max=" << max << " step=" << step;
	TwAddVarRW(bar, name, TW_TYPE_INT32, var, ss.str().c_str());
}

void GUI::addFloatVariable(const char* name, float* var, const char* group, float min, float max, float step) const
{
	group =  strcmp(group, "") == 0? "main" : group;
	std::stringstream ss;
	ss << "group='" << group << "'" << " min=" << min << " max=" << max << " step=" << step;
	TwAddVarRW(bar, name, TW_TYPE_FLOAT, var, ss.str().c_str());
}

void GUI::addDirVariable(const char* name, optix::float3* var, const char* group) const
{
	group =  strcmp(group, "") == 0? "main" : group;
	std::stringstream ss;
	ss << "group='" << group << "'";
	TwAddVarRW(bar, name, TW_TYPE_DIR3F, var, ss.str().c_str());
}

void GUI::addColorVariable(const char* name, optix::float3* var, const char* group) const
{
	group = strcmp(group, "") == 0 ? "main" : group;
	std::stringstream ss;
	ss << "group='" << group << "' colormode=rgb";
	TwAddVarRW(bar, name, TW_TYPE_COLOR3F, var, ss.str().c_str());
}

void GUI::addHDRColorVariable(const char* name, optix::float4* var, const char* group) const
{
    group = strcmp(group, "") == 0 ? "main" : group;
    std::stringstream ss;
    ss << "group='" << group << "' colormode=rgb";
    char buf[256];
    _snprintf_s(buf, sizeof buf, "%s%s", name, ", Color");
    addColorVariable(buf, (optix::float3*)var, ss.str().c_str());
    _snprintf_s(buf, sizeof buf, "%s%s", name, ", Size");

    std::stringstream ss2;
    ss2 << "group='" << group;
    addFloatVariable(buf, (float*)&var[3], ss2.str().c_str());
}

void GUI::addHDRColorVariable(const char* name, optix::float3* var, const char* group) const
{
    group = strcmp(group, "") == 0 ? "main" : group;
    std::stringstream ss;
    ss << "group='" << group;
    char buf[256];
    _snprintf_s(buf, sizeof buf, "%s%s", name, ", R");
    TwAddVarRW(bar, buf, TW_TYPE_FLOAT, &var[0], ss.str().c_str());
    _snprintf_s(buf, sizeof buf, "%s%s", name, ", G");
    TwAddVarRW(bar, buf, TW_TYPE_FLOAT, &var[1], ss.str().c_str());
    _snprintf_s(buf, sizeof buf, "%s%s", name, ", B");
    TwAddVarRW(bar, buf, TW_TYPE_FLOAT, &var[2], ss.str().c_str());
}

void GUI::addCheckBox(const char* name, bool* var, const char* group) const
{
	group =  strcmp(group, "") == 0? "main" : group;
	std::stringstream ss;
	ss << "group='" << group << "'";
	TwAddVarRW(bar, name, TW_TYPE_BOOL32, var, ss.str().c_str());
}

void GUI::addDropdownMenu(const char* name, std::vector<GuiDropdownElement>& values, int* value, const char* group) const
{
	group = strcmp(group, "") == 0 ? "main" : group;
	std::stringstream ss;
	ss << "group='" << group << "'";
	TwType en = TwDefineEnum((std::string(name) + "enum").c_str(), reinterpret_cast<TwEnumVal*>(values.data()), values.size());
	TwAddVarRW(bar, name, en, value, ss.str().c_str());
}

void GUI::addIntVariableCallBack(const char* name, GuiSetVarCallback set_var, GuiGetVarCallback get_var, void* data, const char* group, int min, int max, int step) const
{
	group =  strcmp(group, "") == 0? "main" : group;
	std::stringstream ss;
	ss << "group='" << group << "'" << " min=" << min << " max=" << max << " step=" << step;
	TwAddVarCB(bar, name, TW_TYPE_INT32, set_var, get_var, data, ss.str().c_str());
}

void GUI::addFloatVariableCallBack(const char* name, GuiSetVarCallback set_var, GuiGetVarCallback get_var, void* data, const char* group, float min, float max, float step) const
{
	group =  strcmp(group, "") == 0? "main" : group;
	std::stringstream ss;
	ss << "group='" << group << "'" << " min=" << min << " max=" << max << " step=" << step;
	TwAddVarCB(bar, name, TW_TYPE_FLOAT, set_var, get_var, data, ss.str().c_str());
}



void GUI::addDirVariableCallBack(const char* name, GuiSetVarCallback set_var, GuiGetVarCallback get_var, void* data, const char* group) const
{	
	group =  strcmp(group, "") == 0? "main" : group;
	std::stringstream ss;
	ss << "group='" << group << "'";
	TwAddVarCB(bar, name, TW_TYPE_DIR3F, set_var, get_var, data, ss.str().c_str());
}

void GUI::addCheckBoxCallBack(const char* name, GuiSetVarCallback set_var, GuiGetVarCallback get_var, void* data, const char* group) const
{
	group =  strcmp(group, "") == 0? "main" : group;
	std::stringstream ss;
	ss << "group='" << group << "'";
	TwAddVarCB(bar, name, TW_TYPE_BOOL32, set_var, get_var, data, ss.str().c_str());
}

void GUI::addDropdownMenuCallback(const char* name, std::vector<GuiDropdownElement>& values, GuiSetVarCallback set_var, GuiGetVarCallback get_var, void* data, const char* group) const
{
	group = strcmp(group, "") == 0 ? "main" : group;
	std::stringstream ss;
	ss << "group='" << group << "'";
	TwType en = TwDefineEnum((std::string(name) + "enum").c_str(), reinterpret_cast<TwEnumVal*>(values.data()), values.size());
	TwAddVarCB(bar, name, en, set_var, get_var, data, ss.str().c_str());
}

void GUI::addColorVariableCallback(const char * name, GuiSetVarCallback set_var, GuiGetVarCallback get_var, void * data, const char * group) const
{
	group = strcmp(group, "") == 0 ? "main" : group;
	std::stringstream ss;
	ss << "group='" << group << "' colormode=rgb";
	TwAddVarCB(bar, name, TW_TYPE_COLOR3F, set_var, get_var, data, ss.str().c_str());
}

void GUI::addHDRColorVariableCallback(const char* name, GuiSetVarCallback set_varr, GuiGetVarCallback get_varr, GuiSetVarCallback set_varg, GuiGetVarCallback get_varg, GuiSetVarCallback set_varb, GuiGetVarCallback get_varb, void* data, const char* group) const
{
    group = strcmp(group, "") == 0 ? "main" : group;
    std::stringstream ss;
    ss << "group='" << group;
    char buf[256];
    _snprintf_s(buf, sizeof buf, "%s%s", name, ", R");
    TwAddVarCB(bar, buf, TW_TYPE_FLOAT, set_varr, get_varr, data, ss.str().c_str());
    _snprintf_s(buf, sizeof buf, "%s%s", name, ", G");
    TwAddVarCB(bar, buf, TW_TYPE_FLOAT, set_varg, get_varg, data, ss.str().c_str());
    _snprintf_s(buf, sizeof buf, "%s%s", name, ", B");
    TwAddVarCB(bar, buf, TW_TYPE_FLOAT, set_varb, get_varb, data, ss.str().c_str());
}

void GUI::addHDRColorVariableCallback(const char* name, GuiSetVarCallback set_color, GuiGetVarCallback get_color, GuiSetVarCallback set_scale, GuiGetVarCallback get_scale, void* data, const char* group) const
{
    group = strcmp(group, "") == 0 ? "main" : group;
    std::stringstream ss;
    ss << "group='" << group << "' colormode=rgb";
    char buf[256];
    _snprintf_s(buf, sizeof buf, "%s%s", name, ", Color");
    TwAddVarCB(bar, buf, TW_TYPE_COLOR3F, set_color, get_color, data, ss.str().c_str());

    ss.clear();
    ss << "group='" << group << "' colormode=rgb";
    _snprintf_s(buf, sizeof buf, "%s%s", name, ", Scale");
    TwAddVarCB(bar, buf, TW_TYPE_FLOAT, set_scale, get_scale, data, ss.str().c_str());
}


void GUI::addSeparator() const
{
	TwAddSeparator(bar, nullptr, nullptr);
}

void GUI::setReadOnly(const char* name) const
{
	int val = 1;
	TwSetParam(bar, name, "readonly", TW_PARAM_INT32, 1, &val);
}

void GUI::setReadWrite(const char* name) const
{
	int val = 0;
	TwSetParam(bar, name, "readonly", TW_PARAM_INT32, 1, &val);

}

void GUI::setVisible(const char* name, bool isVisible) const
{
	int val = isVisible;
	TwSetParam(bar, name, "visible", TW_PARAM_INT32, 1, &val);
}

void GUI::addButton(const char* name, GuiButtonCallback callback, void * data, const char* group) const
{
	group =  strcmp(group, "") == 0? "main" : group;
	std::stringstream ss;
	ss << "group='" << group << "'";
	TwAddButton(bar, name, callback, data, ss.str().c_str());
}

void GUI::removeVar(const char* name) const
{
	TwRemoveVar(bar, name);
}

void GUI::draw() const
{
	if (visible)
		TwDraw();
}


