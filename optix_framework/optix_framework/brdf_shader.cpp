#include "brdf_shader.h"
#include "folders.h"
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
        mBRDF = std::move(BRDF::create(context, type));
        std::string path;
        if(type == BRDFType::MERL && Dialogs::openFileDialog(path))
        {
            MERLBRDF* other = dynamic_cast<MERLBRDF*>(mBRDF.get());
            other->set_merl_file(path);
        }
    }

    changed |= mBRDF->on_draw();
    return changed;

}

BRDFShader::BRDFShader(const BRDFShader & other) : Shader(other.info)
{
    context = other.context;
    mBRDF = std::make_unique<BRDF>(*other.mBRDF);
}
