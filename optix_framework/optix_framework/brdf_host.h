#pragma once
#include "host_device_common.h"
#include <memory>
#include "brdf_properties.h"
#include "object_host.h"

class BRDF : public std::enable_shared_from_this<BRDF>
{
public:
    BRDF(optix::Context & ctx, BRDFType::Type type);
    virtual bool on_draw();
    virtual void load(Object& obj);
    virtual BRDFType::Type get_type() { return mType; };
    static std::unique_ptr<BRDF> create(optix::Context & ctx, BRDFType::Type type);
    static bool selector_gui(BRDFType::Type &type, std::string id);

protected:
    optix::Context mContext;
    BRDFType::Type mType;

};

class MERLBRDF : public BRDF
{
public:
    MERLBRDF(optix::Context & ctx, BRDFType::Type type) : BRDF(ctx, type) {}
    ~MERLBRDF() { mMerlBuffer->destroy(); }

    void set_merl_file(std::string file);
    void load(Object& obj) override;

    optix::float3 merl_correction = optix::make_float3(1);
    std::vector<float> data;
    optix::float3 reflectance;
    optix::Buffer mMerlBuffer;
    bool mInit = false;
};

