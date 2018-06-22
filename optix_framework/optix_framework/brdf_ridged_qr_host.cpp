#include "brdf_ridged_qr_host.h"
#include "immediate_gui.h"


void RidgedBRDF::load(MaterialHost &obj)
{
	BRDF::load(obj);
	obj.get_optix_material()["sample_orientation"]->setUint(orientation);
	obj.get_optix_material()["ridge_angle"]->setFloat(ridge_angle);
}

bool RidgedBRDF::on_draw()
{
	BRDF::on_draw();
	bool changed = false;
	const char * const items[3] = { "0 deg","90 deg","180 deg" };
	if (ImmediateGUIDraw::Combo("Orientation", &orientation, items, 3,3))
	{
		changed = true;
	}
	if (ImmediateGUIDraw::SliderFloat("Ridge Angle", &ridge_angle, 0, 90, "%.1f"))
	{
		changed = true;
	}
	return changed;		
}


