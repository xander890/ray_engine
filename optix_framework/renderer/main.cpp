// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011
#include "optix_serialize_utils.h"
#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include <glfw_display.h>
#include "ray_workbench.h"
#include "render_task.h"
#include "scattering_material.h"
#include "sky_model.h"
#include "immediate_gui.h" 
 
#include "hemisphere_partition_utils.h"
namespace
{
	void lower_case(char& x)
  {   
    x = tolower(x); 
  } 

	inline void lower_case_string(std::string& s)
	{
    for_each(s.begin(), s.end(), lower_case);
	}
}

void printUsageAndExit( const std::string& argv0 ) 
{
	std::cerr << "Usage  : " << argv0 << " [options] any_object.obj [another.obj ...]" << std::endl
       << "options: --help            | -h            Print this usage message" << std::endl
       << "         --shader         | -sh <shader>  specify the closest hit program to be used for shading" << std::endl
	   << "options: --rectangle <ox oy w h>     renctangle to render." << std::endl
	   << std::endl;

  exit(0);
}


int main( int argc, char** argv ) 
{
	std::vector<float> r;
	std::vector<int> i;
	std::vector<std::string> filenames;
	std::string filename = "";
	std::string output_file = "rendering.raw";
	bool auto_mode = false;
	int frames = -1;
	float time = -1.0f;
	bool nodisplay = false;
    bool scene_found = false;

	for ( int i = 1; i < argc; ++i )
	{
	std::string arg( argv[i]);
	if (arg == "--no-display")
	{
		nodisplay = true;
	}
	if( arg == "-h" || arg == "--help" )
	{
		printUsageAndExit( argv[0] ); 
	}
	else if (arg == "-o" || arg == "--output")
	{
		if (i == argc - 1)
			printUsageAndExit(argv[0]);
		output_file = argv[++i];
		lower_case_string(output_file);
	}
	else if (arg == "-t" || arg == "--time")
	{
		auto_mode = true;
		time = std::stof(argv[++i]);
	}
	else if (arg == "-f" || arg == "--frames")
	{
		auto_mode = true;
		frames = std::stoi(argv[++i]);
	}
	else
	{
		filename = argv[i];
		std::string file_extension;
		size_t idx = filename.find_last_of('.');
		if(idx < filename.length())
		{
			file_extension = filename.substr(idx, filename.length() - idx);
			lower_case_string(file_extension);
		}

        if(file_extension == ".xml")
        {
            if(scene_found)
            {
                Logger::error << "Only one scene file can be provided on command line." << std::endl;
                exit(2);
            }
            scene_found = true;
            filenames.push_back(filename);
            lower_case_string(filenames.back());
        }

        if(file_extension == ".obj")
		{
			filenames.push_back(filename);
			lower_case_string(filenames.back());
		}
	}
	}

	GLFWDisplay::set_requires_display(!nodisplay);
	GLFWDisplay::init( argc, argv );
	
	std::unique_ptr<RenderTask> task = nullptr;

	if (auto_mode)
	{
		if(frames > 0 && time > 0)
		{
			task = std::make_unique<RenderTaskTimeorFrames>(frames, time, output_file, true);
		}
		else if (frames > 0)
		{
			task = std::make_unique<RenderTaskFrames>(frames, output_file, true);
		}
		else if (time > 0.0f)
		{
			task = std::make_unique<RenderTaskTime>(time, output_file, true);
		}
		else
		{
			Logger::error << "Time or frames not specified" << std::endl;
			exit(2);
		}
	}

	try 
	{
		RayWorkbench * scene_o = new RayWorkbench(filenames);

		if (auto_mode)
		{
			scene_o->set_render_task(task);
			scene_o->start_render_task_on_scene_ready();
		}	

		GLFWDisplay::run( "Optix Renderer", scene_o );
	}
	catch(optix::Exception & e )
	{
        Logger::error<< (e.getErrorString().c_str());
        exit(1);
	}
	return 0;
}
