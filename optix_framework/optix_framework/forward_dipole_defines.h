#pragma once
#include "host_device_common.h"
#include "double_support_optix.h"
#include "float.h"
#include "math.h"
#define M_PI           3.14159265358979323846  /* pi */

#define RND_FUNC_FWD_DIP rnd_tea

/* Reject incoming directions that come from within the actual geometry
* (i.e. w.r.t. the actual local Float3 at the incoming point instead of,
* for instance, the modified tangent plane Float3)? */
#define MTS_FWDSCAT_DIPOLE_REJECT_INCOMING_WRT_TRUE_SURFACE_Float3 true

#define MTS_FWDSCAT_GIVE_REAL_AND_VIRTUAL_SOURCE_EQUAL_SAMPLING_WEIGHT false

#define MTS_FWDSCAT_DEBUG
#define MTS_WITH_CANCELLATION_CHECKS

#ifdef MTS_FWDSCAT_DEBUG
# define FSAssert(x)      optix_assert(x)
# define FSAssertWarn(x)  optix_assert(x)
# define SFSAssert(x)     optix_assert(x)
# define SFSAssertWarn(x) optix_assert(x)
# define SAssert(x)      optix_assert(x)
# define SAssertWarn(x)  optix_assert(x)
#else /* This removes the side-effects from the functions! */
# define FSAssert(x)      ((void) 0)
# define FSAssertWarn(x)  ((void) 0)
# define SFSAssert(x)     ((void) 0)
# define SFSAssertWarn(x) ((void) 0)
# define FSAssert(x)      ((void) 0)
# define FSAssertWarn(x)  ((void) 0)
# define SFSAssert(x)     ((void) 0)
# define SFSAssertWarn(x) ((void) 0)
#endif



#ifdef SINGLE_PRECISION
# define MTS_FWDSCAT_DIRECTION_MIN_MU 1e-3
#else
# define MTS_FWDSCAT_DIRECTION_MIN_MU 1e-4
#endif


using namespace optix;
#define Float double
#define Float2 optix::double2
#define Float3 optix::double3
#define Float3d optix::double3
#define MakeFloat3 optix::make_double3
#define MakeFloat2 optix::make_double2
#define MakeFloat3d optix::make_double3

#define Log(x,y,...) optix_print(y "\n", __VA_ARGS__)
#define SLog Log
#define Epsilon 1e-6

enum TangentPlaneMode {
	EUnmodifiedIncoming,
	EUnmodifiedOutgoing,
	EFrisvadEtAl,
	EFrisvadEtAlWithMeanNormal
};

enum DipoleMode {
	EReal = 1,
	EVirt = 2,
	ERealAndVirt = EReal | EVirt,
};

enum ZvMode {
	EClassicDiffusion, /// As in the original Jensen et al. dipole
	EBetterDipoleZv,   /// As in the better dipole model of d'Eon
	EFrisvadEtAlZv,    /// As in the directional dipole model of Frisvad et al.
};

struct ForwardDipoleMaterial
{
	float sigma_s, sigma_a, mu, m_eta;
};

struct ForwardDipoleProperties
{
	bool rejectInternalIncoming;
	bool reciprocal;
	TangentPlaneMode tangentMode;
	ZvMode zvMode;
	bool useEffectiveBRDF;
	DipoleMode dipoleMode;
};
