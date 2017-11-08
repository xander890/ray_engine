// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011
#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include <GLFWDisplay.h>
#include "render_task.h"
#include "full_bssrdf_generator.h"

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
       << "options: --help           | -h            Print this usage message" << std::endl
       << "         --shader         | -sh <shader>  specify the closest hit program to be used for shading" << std::endl
	   << "options: --rectangle <ox oy w h>     renctangle to render." << std::endl
	   << std::endl;
  
  GLFWDisplay::printUsage();

  exit(0);
}


int main( int argc, char** argv ) 
{
	std::vector<std::string> filenames;
	std::string filename = "";
	std::string output_file = "test.bssrdf";
	std::string config_file = "config.xml";
	//std::map<std::string, std::string> parameters;
	bool auto_mode = false;
	int frames = -1;
	float time = -1.0f;
	bool start_bssrdf_generator = false;
	bool nodisplay = false;

	for ( int i = 1; i < argc; ++i ) 
	{
	std::string arg( argv[i] );
	if (arg == "--no-display")
	{
		nodisplay = true;
	}
	else if (arg == "-o" || arg == "--output")
	{
		if (i == argc - 1)
			printUsageAndExit(argv[0]);
		auto_mode = true;
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
	}
	//if ( filenames.size() == 0 ) 
	//  filenames.push_back(string("./meshes/") + "closed_bunny_vn.obj");
	ConfigParameters::init(config_file);
	GLFWDisplay::setRequiresDisplay(!nodisplay);
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
		SampleScene * scene  = new FullBSSRDFGenerator(config_file.c_str(), auto_mode);
		if(auto_mode)
		{
			((FullBSSRDFGenerator*)scene)->set_render_task(task);
		}

		GLFWDisplay::run( "Optix Renderer", scene );
	} 
	catch(optix::Exception & e )
	{
	Logger::error<<  (e.getErrorString().c_str());
	exit(1);
	}
	return 0;
}
