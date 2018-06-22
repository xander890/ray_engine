#include "brdf_shader.h"
#include "object_host.h"
#include "merl_common.h"
#include <image_loader.h>
#include "material_host.h"
#include "file_dialogs.h"
#include "merl_host.h"


void BRDFShader::initialize_shader(optix::Context context)
{
    Shader::initialize_shader(context);
    mBRDF = BRDF::create(context, BRDFType::LAMBERTIAN);
}

void BRDFShader::initialize_material(MaterialHost &object)
{
    Shader::initialize_material(object);
    mBRDF->load(object);
}

void BRDFShader::pre_trace_mesh(Object& object)
{
    Shader::pre_trace_mesh(object);
}

bool BRDFShader::on_draw()
{
    static BRDFType::Type type = mBRDF->get_type();
    bool changed = false;

    if(BRDF::selector_gui(type, ""))
    {
        changed = true;
        mBRDF.reset();
        mBRDF = BRDF::create(context, type);
        std::string path;
        if(type == BRDFType::MERL && Dialogs::openFileDialog(path))
        {
            MERLBRDF* other = dynamic_cast<MERLBRDF*>(mBRDF.get());
            other->load_brdf_file(path);
        }
    }

    mBRDF->on_draw();
    return changed;

}

BRDFShader::BRDFShader(const BRDFShader & other) : Shader(other.info)
{
    context = other.context;
    mBRDF = std::make_unique<BRDF>(*other.mBRDF);
}
