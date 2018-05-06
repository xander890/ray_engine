#pragma once

#include <immediate_gui.h>
#include "bssrdf_host.h"
#include "bssrdf_loader.h"
#include "empirical_bssrdf_common.h"

class BSSRDFImporter;

class EmpiricalBSSRDF : public BSSRDF
{
public:
	EmpiricalBSSRDF(optix::Context & ctx);
    ~EmpiricalBSSRDF() {}
	void load(const float relative_ior, const ScatteringMaterialProperties &props) override;
	bool on_draw() override;

private:
    void prepare_buffers();
    bool mInitialized = false;
    EmpiricalParameterBuffer mParameterBuffers; // One buffer per parameter
    optix::Buffer mParameterSizeBuffer;
    EmpiricalDataBuffer mDataBuffers;
    std::string mBSSRDFFile;
    std::unique_ptr<BSSRDFImporter> mBSSRDFLoader = nullptr;
    std::unique_ptr<BSSRDFParameterManager> mManager = nullptr;

	float mCorrection = 1.0f;
	unsigned int mInterpolation = 0;
    EmpiricalBSSRDFNonPlanarity::Type mNonPlanarSurfacesHandles = EmpiricalBSSRDFNonPlanarity::UNCHANGED;

	friend class cereal::access;
	template<class Archive>
	void serialize(Archive & archive)
	{
		// FIXME handle load save properly.
		archive(
				cereal::base_class<BSSRDF>(this)
		);
	}
};
