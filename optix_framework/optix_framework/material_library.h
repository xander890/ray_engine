#pragma once

#include <optix_world.h>
#include <string>
#include <map>
#include "math_helpers.h"
#include "structs.h"

// Optix-friendly wrapper for the Medium and MPML readers from the GEL library
// This wrapper is made not to pollute our code with undesired CGLA references 

class Medium;

class MPMLInterface
{
public:
	MPMLInterface() : med_in(nullptr), med_out(nullptr) { }

	std::string name;
	MPMLMedium* med_in;
	MPMLMedium* med_out;
};


class MaterialLibrary
{
public:
	static std::map<std::string, MPMLMedium> media;
	static std::map<std::string, Medium> full_media;
	static std::map<std::string, MPMLInterface> interfaces;
	static void load(const char * mpml_path);

private:
	static void convert_and_store(Medium m);
};


void convert_mediums(Medium & medium, MPMLMedium & new_medium);

void load_mpml(const std::string & filename, 
			   std::map<std::string, MPMLMedium>& media, 
			   std::map<std::string, MPMLInterface>& interface_map);

