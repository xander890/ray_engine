#pragma once
#include <exception>
#include <vector>

typedef struct CTwBar TwBar;
typedef void (__stdcall * GuiSetVarCallback)(const void *value, void *clientData);
typedef void(__stdcall * GuiGetVarCallback)(void *value, void *clientData);
typedef void(__stdcall * GuiButtonCallback)(void *clientData);
#define GUI_CALL          __stdcall

namespace optix{
    struct float3;
    struct float4;
}

struct GuiDropdownElement
{
	int           Value;
	const char *  Label;
};

class GUI
{
public:
	GUI(const char* name, int window_width, int window_height);

	virtual ~GUI();


	bool keyPressed(unsigned char key, int x, int y);
	// Use this to add additional keys. Some are already handled but
	// can be overridden.  Should return true if key was handled, false otherwise.
	bool mousePressed(int button, int state, int x, int y);
	bool mouseMoving(int x, int y);
	static void setWindowSize(int x, int y);

	void draw() const;

	void toggleVisibility() { visible = !visible; }
	bool isVisible() const
	{
		return visible;
	}
	
	void addIntVariable(const char * name, int * var, const char * group = "", int min = 0, int max = INT_MAX, int step = 1) const;
	void addFloatVariable(const char * name, float * var, const char * group = "", float min = 0.0f, float max = FLT_MAX, float step = 0.001f) const;
	void addDirVariable(const char * name, optix::float3 * var, const char * group = "") const;
	void addColorVariable(const char * name, optix::float3 * var, const char * group = "") const;
    void addHDRColorVariable(const char * name, optix::float4 * var, const char * group = "") const;
    void addHDRColorVariable(const char * name, optix::float3 * var, const char * group = "") const;

	void addCheckBox(const char * name, bool * var, const char * group = "") const;
	void addDropdownMenu(const char * name, std::vector<GuiDropdownElement> & values, int * value, const char * group = "") const;
    void addIntVariableCallBack(const char * name, GuiSetVarCallback set_var, GuiGetVarCallback get_var, void * data, const char * group = "", int min = 0, int max = INT_MAX, int step = 1) const;
	void addFloatVariableCallBack(const char * name, GuiSetVarCallback set_var, GuiGetVarCallback get_var, void * data, const char * group = "", float min = 0.0f, float max = FLT_MAX, float step = 0.001f) const;
	void addDirVariableCallBack(const char * name, GuiSetVarCallback set_var, GuiGetVarCallback get_var, void * data, const char * group = "") const;
	void addCheckBoxCallBack(const char * name, GuiSetVarCallback set_var, GuiGetVarCallback get_var, void * data, const char * group = "") const;
	void addDropdownMenuCallback(const char * name, std::vector<GuiDropdownElement> & values, GuiSetVarCallback set_var, GuiGetVarCallback get_var, void * data, const char * group = "") const;
	void addColorVariableCallback(const char * name, GuiSetVarCallback set_var, GuiGetVarCallback get_var, void * data, const char * group = "") const;
    void addHDRColorVariableCallback(const char * name, GuiSetVarCallback set_varr, GuiGetVarCallback get_varr, GuiSetVarCallback set_varg, GuiGetVarCallback get_varg, GuiSetVarCallback set_varb, GuiGetVarCallback get_varb, void * data, const char * group = "") const;
    void addHDRColorVariableCallback(const char * name, GuiSetVarCallback set_color, GuiGetVarCallback get_color, GuiSetVarCallback set_scale, GuiGetVarCallback get_scale, void * data, const char * group = "") const;

	void addSeparator() const;
	void addButton(const char* name, GuiButtonCallback callback, void * data, const char* group = "") const;
	void setReadOnly(const char* name) const;
	void setReadWrite(const char* name) const;
	void setVisible(const char * name, bool isVisible) const;
	void removeVar(const char * name) const;

    void linkGroups(const char * parent, const char * child);

private:
	TwBar * bar;
	bool visible = true;
    std::string barname;

};

