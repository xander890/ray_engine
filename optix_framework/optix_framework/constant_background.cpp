#include "constant_background.h"
#include "folders.h"
#include "host_device_common.h"
#include "immediate_gui.h"

void ConstantBackground::init()  
{
    MissProgram::init();
	mContext["bg_color"]->setFloat(mBackgroundColor);
}

void ConstantBackground::load()  
{
    MissProgram::load();
    mContext["bg_color"]->setFloat(mBackgroundColor);
}

bool ConstantBackground::on_draw()
{
	bool changed = false;
	if (ImmediateGUIDraw::TreeNode("Constant Color"))
	{
		changed |= ImmediateGUIDraw::ColorEdit3("Color", (float*)&mBackgroundColor, ImGuiColorEditFlags_NoAlpha);
		ImGui::TreePop();
	}
	return changed;
}

bool ConstantBackground::get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program)
{
    const char * prog = ray_type ==  RayType::SHADOW ? "miss_shadow" : "miss";
    program = ctx->createProgramFromPTXFile(get_path_ptx("constant_background.cu"), prog);
    return true;
}