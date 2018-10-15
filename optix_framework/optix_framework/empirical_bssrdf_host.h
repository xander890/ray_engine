#pragma once

#include <immediate_gui.h>
#include "bssrdf_host.h"
#include "empirical_bssrdf.h"
#include "empirical_bssrdf_common.h"

/*
 * Specialization of BSSRDF class to represent an empirical BSSRDF. The empirical BSSRDF consist of a series of axes values (albedo, eta, g, theta_s, r, theta_i, etc.) and a table of corresponding values. This class handles reading  the BSSRDF and uploading it to the GPU in a appropriate fashion.
 */

class BSSRDFImporter;

class EmpiricalBSSRDFImpl : public BSSRDF
{
public:
	EmpiricalBSSRDFImpl(optix::Context & ctx);
    ~EmpiricalBSSRDFImpl() {}
	void load(const optix::float3 &relative_ior, const ScatteringMaterialProperties &props) override;
	bool on_draw() override;

private:
    void prepare_buffers();
    bool mInitialized = false;
    EmpiricalParameterBuffer mParameterBuffers; // One buffer per parameter
    optix::Buffer mParameterSizeBuffer;
    EmpiricalDataBuffer mDataBuffers;
    std::string mBSSRDFFile;
    std::unique_ptr<EmpiricalBSSRDF> mBSSRDFManager = nullptr;

	float mCorrection = 1.0f;
	unsigned int mInterpolation = 0;
    EmpiricalBSSRDFNonPlanarity::Type mNonPlanarSurfacesHandles = EmpiricalBSSRDFNonPlanarity::UNCHANGED;

	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		// TODO handle load save properly.
		archive(
				cereal::base_class<BSSRDF>(this)
		);
	}
};
