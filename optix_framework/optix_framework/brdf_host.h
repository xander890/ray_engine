#pragma once
#include "optix_serialize.h"
#include "host_device_common.h"
#include <memory>
#include "brdf_properties.h"
#include "host_material.h"

class Object;

class BRDF
{
public:
    BRDF(optix::Context & ctx, BRDFType::Type type);
    BRDF(const BRDF& other);
    virtual bool on_draw();
    virtual void load(MaterialHost &obj);
    virtual BRDFType::Type get_type() { return mType; };
    static std::unique_ptr<BRDF> create(optix::Context & ctx, BRDFType::Type type);
    static bool selector_gui(BRDFType::Type &type, std::string id);

protected:
    optix::Context mContext;
    BRDFType::Type mType;
    BRDF() {};

private:
    friend class cereal::access;

    void load(cereal::XMLInputArchiveOptix & archive)
    {
        mContext = archive.get_context();
        std::string type;
        archive(cereal::make_nvp("type", type));
        mType = BRDFType::to_enum(type);
    }

    template<class Archive>
    void save(Archive & archive) const
    {
        archive(cereal::make_nvp("type", BRDFType::to_string(mType)));
    }

};

CEREAL_CLASS_VERSION(BRDF, 0)
CEREAL_REGISTER_TYPE(BRDF)

class MERLBRDF : public BRDF
{
public:
    MERLBRDF(optix::Context & ctx, BRDFType::Type type) : BRDF(ctx, type)
    {

    }
    MERLBRDF(const MERLBRDF& other);
    MERLBRDF& operator=(const MERLBRDF& other);

    ~MERLBRDF() { mMerlBuffer->destroy(); }

    void set_merl_file(std::string file);
    void load(MaterialHost &obj) override;
    bool on_draw() override;

    optix::float3 merl_correction = optix::make_float3(1);
    std::vector<float> data;
    optix::float3 reflectance;
    optix::Buffer mMerlBuffer;
    std::string mName;
    bool mInit = false;

private:
    void init();

    friend class cereal::access;
    MERLBRDF() : BRDF() {};

    void load(cereal::XMLInputArchiveOptix & archive)
    {
        int size;
        archive( cereal::base_class<BRDF>(this),
                cereal::make_nvp("name", mName),
                cereal::make_nvp("reflectance", reflectance),
                cereal::make_nvp("merl_correction", merl_correction),
                cereal::make_nvp("merl_size", size)
        );
        data.resize(size);
        archive.loadBinaryValue(data.data(), size * sizeof(float), "merl_data");
    }

    void save(cereal::XMLOutputArchiveOptix & archive) const
    {
        archive( cereal::base_class<BRDF>(this),
                cereal::make_nvp("name", mName),
                cereal::make_nvp("reflectance", reflectance),
                cereal::make_nvp("merl_correction", merl_correction),
                cereal::make_nvp("merl_size", data.size())
        );

        archive.saveBinaryValue(data.data(), data.size() * sizeof(float), "merl_data");
    }

};

CEREAL_CLASS_VERSION(MERLBRDF, 0)
CEREAL_REGISTER_TYPE(MERLBRDF)
CEREAL_REGISTER_POLYMORPHIC_RELATION(BRDF, MERLBRDF)