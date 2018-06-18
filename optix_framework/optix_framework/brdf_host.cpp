//
// Created by alcor on 5/2/18.
//

#include "brdf_host.h"
#include "merl_common.h"
#include "brdf_utils.h"
#include "string_utils.h"
#include "immediate_gui.h"
#include "object_host.h"

std::unique_ptr<BRDF> BRDF::create(optix::Context &ctx, BRDFType::Type type)
{
    switch (type)
    {
        case BRDFType::LAMBERTIAN:return std::unique_ptr<BRDF>(new BRDF(ctx, BRDFType::LAMBERTIAN));
        case BRDFType::TORRANCE_SPARROW :return std::unique_ptr<BRDF>(new BRDF(ctx, BRDFType::TORRANCE_SPARROW));
        case BRDFType::MERL :return std::unique_ptr<BRDF>(new MERLBRDF(ctx, BRDFType::MERL));
        case BRDFType::GGX :return std::unique_ptr<BRDF>(new BRDF(ctx, BRDFType::GGX));
    }
	return nullptr; 
}

bool BRDF::on_draw()
{
    return false;
}

BRDF::BRDF(optix::Context &ctx, BRDFType::Type type) : mType(type), mContext(ctx)
{

}

void BRDF::load(MaterialHost &obj)
{
    obj.get_optix_material()["selected_brdf"]->setUserData(sizeof(BRDFType::Type), &mType);
}


void MERLBRDF::set_merl_file(std::string file)
{
    read_brdf_f("", file, data);
    mName = file.substr(file.find_last_of("/\\") + 1);
    reflectance = integrate_brdf(data, 100000);
}

void MERLBRDF::load(MaterialHost &obj)
{
    BRDF::load(obj);

    if(!mInit)
        init();

    obj.get_optix_material()["merl_brdf_multiplier"]->setFloat(merl_correction);
    BufPtr1D<float> ptr = BufPtr1D<float>(mMerlBuffer->getId());
    obj.get_optix_material()["merl_brdf_buffer"]->setUserData(sizeof(BufPtr1D<float>), &ptr);
}

void MERLBRDF::init()
{
    if(mMerlBuffer.get() == nullptr)
    {
        mMerlBuffer = mContext->createBuffer(RT_BUFFER_INPUT);
        mMerlBuffer->setFormat(RT_FORMAT_FLOAT);
        mMerlBuffer->setSize(data.size());
    }
    reflectance = integrate_brdf(data, 100000);
    void* b = mMerlBuffer->map();
    memcpy(b, data.data(), data.size() * sizeof(float));
    mMerlBuffer->unmap();
    mInit = true;
}

MERLBRDF::MERLBRDF(const MERLBRDF &other) : BRDF(other)
{
    merl_correction = optix::make_float3(1);
    data = other.data;
    reflectance = other.reflectance;
    mName = other.mName;
}

MERLBRDF &MERLBRDF::operator=(const MERLBRDF &other)
{
    mContext = other.mContext;
    mType = other.mType;
    merl_correction = optix::make_float3(1);
    data = other.data;
    reflectance = other.reflectance;
    mName = other.mName;
	return *this;
}

bool MERLBRDF::on_draw()
{
    ImmediateGUIDraw::TextWrapped("%f %f %f \n", reflectance.x, reflectance.y, reflectance.z);
    return false;
}


bool BRDF::selector_gui(BRDFType::Type & type, std::string id)
{
    std::string dipoles = "";
    BRDFType::Type t = BRDFType::first();
    do
    {
        dipoles += prettify(BRDFType::to_string(t)) + '\0';
        t = BRDFType::next(t);
    } while (t != BRDFType::NotValidEnumItem);

#define ID_STRING(x,id) (std::string(x) + "##" + id + x).c_str()
    if (ImmediateGUIDraw::Combo(ID_STRING("Dipole", id), (int*)&type, dipoles.c_str(), BRDFType::count()))
    {
        return true;
    }
    return false;
#undef ID_STRING
}

BRDF::BRDF(const BRDF &other)
{
    mContext = other.mContext;
    mType = other.mType;
}
