#pragma once
#include "brdf_common.h"
#include "material_host.h"
#include "optix_serialize_utils.h"
#include <memory>

/*
 * Class representing a BRDF, either analytical or empirical. 
 * The BRDF is loaded into the GPU through these class or their derivatives.
 * See brdf.h header for the usage of the individual BRDFs.
 */
class BRDF
{
public:
	BRDF(optix::Context & ctx, BRDFType::Type type);
    BRDF(const BRDF& other);
	virtual ~BRDF();

	// Draws the BRDF on the immediate GUI.
    virtual bool on_draw();

	// Loads the BRDF into a specific material.
	virtual void load(MaterialHost &obj);

	// Returns the BRDF type.
	virtual BRDFType::Type get_type() { return mType; };

	// Use this function to create a BRDF of a specific type. See the BRDFType enum for details on the
	// available ones.
	static std::unique_ptr<BRDF> create(optix::Context & ctx, BRDFType::Type type);

	// Immediate style gui to select a BRDF.
    static bool selector_gui(BRDFType::Type &type, std::string id);

protected:
	BRDF() = default; // Do not use in derived classes. It is for convenience of serialization functions.
	optix::Context mContext = nullptr;
    BRDFType::Type mType = BRDFType::LAMBERTIAN;

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

