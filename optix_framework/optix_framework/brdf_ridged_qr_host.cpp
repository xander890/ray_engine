#include "brdf_ridged_qr_host.h"
#include "immediate_gui.h"


void RidgedBRDF::load(MaterialHost &obj)
{
	BRDF::load(obj);
	obj.get_optix_material()["ridge_angle"]->setFloat(ridge_angle);
}

bool RidgedBRDF::on_draw()
{
	BRDF::on_draw();
	bool changed = false;
	if (ImmediateGUIDraw::SliderFloat("Ridge Angle", &ridge_angle, 0, 90, "%.1f"))
	{
		changed = true;
	}
	return changed;		
}


