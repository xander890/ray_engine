#pragma once
#include "miss_program.h"
#include "optix_serialize_utils.h"

/*
 * Simple miss program with a constant color.
 */
class ConstantBackground : public MissProgram
{
public:
    ConstantBackground(optix::Context & ctx, const optix::float3& bg = optix::make_float3(0.5f)) : MissProgram(ctx), mBackgroundColor(bg) {}
	virtual ~ConstantBackground() = default;

    virtual void init() override;
    virtual void load() override;
	virtual bool on_draw() override;
private:
    virtual bool get_miss_program(unsigned int ray_type, optix::Context & ctx, optix::Program & program) override;
    optix::float3 mBackgroundColor;

	ConstantBackground() : MissProgram() {}

    friend class cereal::access;

	void load(cereal::XMLInputArchiveOptix & archive) {
		mContext = archive.get_context();
		archive(cereal::virtual_base_class<MissProgram>(this), cereal::make_nvp("background_color", mBackgroundColor));
	}

    template<class Archive>
    void save(Archive & archive) const    {
        archive(cereal::virtual_base_class<MissProgram>(this), cereal::make_nvp("background_color", mBackgroundColor));
    }
};

CEREAL_CLASS_VERSION(ConstantBackground, 0)
CEREAL_REGISTER_TYPE(ConstantBackground)
CEREAL_REGISTER_POLYMORPHIC_RELATION(MissProgram, ConstantBackground)