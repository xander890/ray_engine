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
	BRDF() {} // Do not use in derived classes. It is for convenience of serialization functions.
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

/*
 * Class representing a MERL material. Material is loaded into buffers via this class, then sampled at runtime.
 */
class MERLBRDF : public BRDF
{
public:
    MERLBRDF(optix::Context & ctx, BRDFType::Type type) : BRDF(ctx, type)
    {

    }
    MERLBRDF(const MERLBRDF& other);
    MERLBRDF& operator=(const MERLBRDF& other);

	~MERLBRDF();

	// Loads data from a .binary MERL brdf file into the class.
    void load_brdf_file(std::string file);

	// Loads the data into a specific material.
	void load(MaterialHost &obj) override;

    bool on_draw() override;

private:
    void init();
	MERLBRDF() : BRDF() {}

	// Serialization classes
    friend class cereal::access;
    void load(cereal::XMLInputArchiveOptix & archive)
    {
        int size;
        archive( cereal::base_class<BRDF>(this),
                cereal::make_nvp("name", mName),
                cereal::make_nvp("mReflectance", mReflectance),
                cereal::make_nvp("merl_correction", mCorrection),
                cereal::make_nvp("merl_size", size)
        );
        mData.resize(size);
        archive.loadBinaryValue(mData.data(), size * sizeof(float), "merl_data");
    }

    void save(cereal::XMLOutputArchiveOptix & archive) const
    {
        archive( cereal::base_class<BRDF>(this),
                cereal::make_nvp("name", mName),
                cereal::make_nvp("mReflectance", mReflectance),
                cereal::make_nvp("merl_correction", mCorrection),
                cereal::make_nvp("merl_size", mData.size())
        );

        archive.saveBinaryValue(mData.data(), mData.size() * sizeof(float), "merl_data");
    }
	bool mInit = false;
	optix::float3 mCorrection = optix::make_float3(1);
	std::vector<float> mData;
	optix::float3 mReflectance = optix::make_float3(0);
	optix::Buffer mMerlBuffer = nullptr;
	std::string mName = "";
};

CEREAL_CLASS_VERSION(MERLBRDF, 0)
CEREAL_REGISTER_TYPE(MERLBRDF)
CEREAL_REGISTER_POLYMORPHIC_RELATION(BRDF, MERLBRDF)