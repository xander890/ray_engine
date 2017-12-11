#include "material_library.h"
#include "optprops/Medium.h"
#include "optprops/load_mpml.h"
#include "optprops/Interface.h"
#include <optix_math.h>
#include "optprops/glass.h"
#include "logger.h"

using namespace std;
using namespace optix;

typedef map<string, Medium>::iterator mat_it;
typedef map<string, Interface>::iterator int_it;

optix::float3 color_to_float3(Color<double> & color)
{
	return optix::make_float3(static_cast<float>(color[0]),static_cast<float>(color[1]),static_cast<float>(color[2]));
}

void convert_mediums(Medium & medium, MPMLMedium & new_medium)
{
	medium.fill_rgb_data();
	new_medium.name =				medium.name;
	new_medium.absorption =			color_to_float3(medium.get_absorption(rgb));
	new_medium.albedo =				color_to_float3(medium.get_albedo(rgb));
	new_medium.asymmetry =			color_to_float3(medium.get_asymmetry(rgb));
	//new_medium.emission =			color_to_float3(medium.get_emission(rgb));
	new_medium.extinction =			color_to_float3(medium.get_extinction(rgb));
	new_medium.reduced_albedo =		color_to_float3(medium.get_reduced_alb(rgb));
	new_medium.reduced_extinction = color_to_float3(medium.get_reduced_ext(rgb));
	new_medium.reduced_scattering = color_to_float3(medium.get_reduced_sca(rgb));
	new_medium.scattering =			color_to_float3(medium.get_scattering(rgb));
	Color<std::complex<double>> ior =	medium.get_ior(rgb);
    new_medium.ior_real = optix::make_float3((float)ior[0].real(), (float)ior[1].real(), (float)ior[2].real());
    new_medium.ior_imag = optix::make_float3((float)ior[0].imag(), (float)ior[1].imag(), (float)ior[2].imag());
}

void load_mpml(const string & filename, map<string, MPMLMedium>& media, map<string, Medium>& full_media, map<string, MPMLInterface>& interface_map)
{
	map<string, Medium> media_old;
	map<string, Interface> interfaces;
	load_mpml(filename,media_old,interfaces);

	for(mat_it iterator = media_old.begin(); iterator != media_old.end(); iterator++) {
		string name = iterator->first;

		Medium med = iterator->second;
		med.fill_rgb_data();
		MPMLMedium * mat = new MPMLMedium();
		convert_mediums(med,*mat);
		Logger::debug << "Loading media: " << name << endl;
		media[name] = *mat;
		full_media[name] = med;
	}

	for(int_it iterator_interf = interfaces.begin(); iterator_interf != interfaces.end(); iterator_interf++) {
		string name = iterator_interf->first;
		Interface intef = iterator_interf->second;
		MPMLInterface * new_interface = new MPMLInterface();
		new_interface->name = name;
		if(intef.med_in && media.count(intef.med_in->name) != 0)
			new_interface->med_in = &(media.at(intef.med_in->name));
		if(intef.med_out && media.count(intef.med_out->name) != 0)
			new_interface->med_out = &(media.at(intef.med_out->name));
		interface_map[name] = *new_interface;
	}
}

map<string, MPMLInterface> MaterialLibrary::interfaces = map<string, MPMLInterface>();
map<string, MPMLMedium> MaterialLibrary::media = map<string, MPMLMedium>();
map<string, Medium> MaterialLibrary::full_media = map<string, Medium>();

void MaterialLibrary::convert_and_store(Medium m)
{
	MPMLMedium * new_medium = new MPMLMedium();
	convert_mediums(m, *new_medium);
	media[m.name] = *new_medium;
	full_media[m.name] = m; 
}

void MaterialLibrary::load(const char * mpml_path)
{
	load_mpml(mpml_path,media, full_media, interfaces);
	Medium air;
	air.get_ior(mono).resize(1);
	air.get_ior(mono)[0] = complex<double>(1.0, 0.0);
	air.fill_rgb_data();
	air.name = "air";
	air.turbid = false;
	MPMLMedium * air_converted = new MPMLMedium();
	convert_mediums(air,*air_converted);
	media["air"] = *air_converted;

	convert_and_store(deep_crown_glass());	
	convert_and_store(crown_glass());
	convert_and_store(crown_flint_glass());
	convert_and_store(light_flint_glass());
	convert_and_store(dense_barium_flint_glass());
	convert_and_store(dense_flint_glass());
}

