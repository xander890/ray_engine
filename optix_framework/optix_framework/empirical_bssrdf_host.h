#pragma once

#include <immediate_gui.h>
#include "bssrdf_host.h"
#include "bssrdf_loader.h"
#include "empirical_bssrdf_utils.h"

class BSSRDFLoader;

class EmpiricalBSSRDF : public BSSRDF
{
public:
	EmpiricalBSSRDF(optix::Context & ctx);
    ~EmpiricalBSSRDF() {}
	void load(const float relative_ior, const ScatteringMaterialProperties &props) override;
	void on_draw() override {
		if(ImmediateGUIDraw::InputFloat("Correction" ,&mCorrection))
        {
            mContext["empirical_bssrdf_correction"]->setFloat(mCorrection);
        }

	}

private:
    void prepare_buffers();
    bool mInitialized = false;
    EmpiricalParameterBuffer mParameterBuffers; // One buffer per parameter
    optix::Buffer mParameterSizeBuffer;
    EmpiricalDataBuffer mDataBuffers;
    std::string mBSSRDFFile;
    std::unique_ptr<BSSRDFLoader> mBSSRDFLoader = nullptr;
    std::unique_ptr<BSSRDFParameterManager> mManager = nullptr;
	float mCorrection = DEFAULT_EMPIRICAL_CORRECTION;
};
