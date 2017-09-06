#include "constant_background.h"
#include "folders.h"
#include "host_device_common.h"
#include "immediate_gui.h"

void ConstantBackground::init(optix::Context & ctx)  
{
    MissProgram::init(ctx);
    ctx["bg_color"]->setFloat(background_color);
}

void ConstantBackground::set_into_gpu(optix::Context & ctx)  
{
    MissProgram::set_into_gpu(ctx);
    ctx["bg_color"]->setFloat(background_color);
}

void ConstantBackground::on_draw()
{
	if (ImmediateGUIDraw::TreeNode("Constant Color"))
	{
		ImmediateGUIDraw::ColorEdit3("Color", (float*)&background_color, ImGuiColorEditFlags_NoAlpha);
		ImGui::TreePop();
	}
	
}

bool ConstantBackground::get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program)
{
    const char * prog = ray_type == RAY_TYPE_SHADOW ? "miss_shadow" : "miss";
    program = ctx->createProgramFromPTXFile(get_path_ptx("constant_background.cu"), prog);
    return true;
}