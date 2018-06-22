#include "brdf_host.h"
#include "merl_common.h"
#include "merl_utils.h"
#include "string_utils.h"
#include "immediate_gui.h"
#include "object_host.h"
#include "merl_host.h"
#include "brdf_ridged_qr_host.h"

std::unique_ptr<BRDF> BRDF::create(optix::Context &ctx, BRDFType::Type type)
{
    switch (type)
    {
        case BRDFType::LAMBERTIAN:return std::make_unique<BRDF>(ctx, BRDFType::LAMBERTIAN);
        case BRDFType::TORRANCE_SPARROW :return std::make_unique<BRDF>(ctx, BRDFType::TORRANCE_SPARROW);
        case BRDFType::MERL :return std::make_unique<MERLBRDF>(ctx, BRDFType::MERL);
        case BRDFType::GGX :return std::make_unique<BRDF>(ctx, BRDFType::GGX);
		case BRDFType::BECKMANN: return std::make_unique<BRDF>(ctx, BRDFType::BECKMANN);
		case BRDFType::QR_RIDGED:return std::make_unique<RidgedBRDF>(ctx, BRDFType::QR_RIDGED);
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
    if (ImmediateGUIDraw::Combo(ID_STRING("BRDF", id), (int*)&type, dipoles.c_str(), BRDFType::count()))
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

BRDF::~BRDF()
{
}
