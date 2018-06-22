#include "merl_host.h"

#include "merl_common.h"
#include "merl_utils.h"
#include "string_utils.h"
#include "immediate_gui.h"
#include "object_host.h"

void MERLBRDF::load_brdf_file(std::string file)
{
	read_brdf_f("", file, mData);
	mName = file.substr(file.find_last_of("/\\") + 1);
	mReflectance = integrate_brdf(mData, 100000);
}

void MERLBRDF::load(MaterialHost &obj)
{
	BRDF::load(obj);

	if (!mInit)
		init();

	obj.get_optix_material()["merl_brdf_multiplier"]->setFloat(mCorrection);
	BufPtr1D<float> ptr = BufPtr1D<float>(mMerlBuffer->getId());
	obj.get_optix_material()["merl_brdf_buffer"]->setUserData(sizeof(BufPtr1D<float>), &ptr);
}

void MERLBRDF::init()
{
	if (mMerlBuffer.get() == nullptr)
	{
		mMerlBuffer = mContext->createBuffer(RT_BUFFER_INPUT);
		mMerlBuffer->setFormat(RT_FORMAT_FLOAT);
		mMerlBuffer->setSize(mData.size());
	}
	mReflectance = integrate_brdf(mData, 100000);
	void* b = mMerlBuffer->map();
	memcpy(b, mData.data(), mData.size() * sizeof(float));
	mMerlBuffer->unmap();
	mInit = true;
}

MERLBRDF::MERLBRDF(const MERLBRDF &other) : BRDF(other)
{
	mCorrection = optix::make_float3(1);
	mData = other.mData;
	mReflectance = other.mReflectance;
	mName = other.mName;
}

MERLBRDF &MERLBRDF::operator=(const MERLBRDF &other)
{
	mContext = other.mContext;
	mType = other.mType;
	mCorrection = optix::make_float3(1);
	mData = other.mData;
	mReflectance = other.mReflectance;
	mName = other.mName;
	return *this;
}

MERLBRDF::~MERLBRDF()
{
	mMerlBuffer->destroy();
}

bool MERLBRDF::on_draw()
{
	BRDF::on_draw();
	ImmediateGUIDraw::TextWrapped("%f %f %f \n", mReflectance.x, mReflectance.y, mReflectance.z);
	return false;
}