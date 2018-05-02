#include "glossy.h"
#include "folders.h"
#include "parameter_parser.h"
#include "object_host.h"
#include "brdf_utils.h"
#include "merl_common.h"
#include <ImageLoader.h>
#include "host_material.h"
#include "scattering_material.h"
#include "dialogs.h"
#include <algorithm>

void BRDFShader::initialize_shader(optix::Context context)
{
    Shader::initialize_shader(context);
    mBRDF = BRDF::create(context, BRDFType::LAMBERTIAN);
}

void BRDFShader::initialize_mesh(Object& object)
{
    Shader::initialize_mesh(object);
    mBRDF->load(object);
}

void BRDFShader::pre_trace_mesh(Object& object)
{
    Shader::pre_trace_mesh(object);
    mBRDF->load(object);
}

bool BRDFShader::on_draw()
{
    static BRDFType::Type type = mBRDF->get_type();
    bool changed = false;

    if(BRDF::selector_gui(type, ""))
    {
        changed = true;
        mBRDF.reset();
        mBRDF = std::move(BRDF::create(context, type));
        std::string path;
        if(type == BRDFType::MERL && Dialogs::openFileDialog(path))
        {
            std::dynamic_pointer_cast<MERLBRDF>(mBRDF)->set_merl_file(path);
        }
    }

    mBRDF->on_draw();
    return changed;

}
