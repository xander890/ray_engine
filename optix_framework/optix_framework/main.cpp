// 02576 OptiX Rendering Framework
// Written by Jeppe Revall Frisvad, 2011
// Copyright (c) DTU Informatics 2011

#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>
#include <GLUTDisplay.h>
#include "obj_scene.h"
#include "render_task.h"

using namespace std;
using namespace optix;

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

void printUsageAndExit( const string& argv0 ) 
{
  cerr << "Usage  : " << argv0 << " [options] any_object.obj [another.obj ...]" << endl
       << "options: --help           | -h            Print this usage message" << endl
       << "         --shader         | -sh <shader>  specify the closest hit program to be used for shading" << endl
	   << "options: --rectangle <ox oy w h>     renctangle to render." << endl
	   << endl;
  
  GLUTDisplay::printUsage();

  exit(0);
}

int main( int argc, char** argv ) 
{
  vector<string> filenames;
  string filename = "";
  string shadername = "";
  string output_file = "rendering.raw";
  string config_file = "config.xml";
  bool accel_caching_on = false;
  int4 rendering_rect = make_int4(-1);
  //std::map<std::string, std::string> parameters;
  bool auto_mode = false;
  int frames = 0;
  std::unique_ptr<RenderTask> task = make_unique<RenderTask>();
  task->close_program_on_exit = true;
  std::string material_override_mtl = "";

  for ( int i = 1; i < argc; ++i ) 
  {
    string arg( argv[i] );
    if( arg == "-h" || arg == "--help" ) 
    {
      printUsageAndExit( argv[0] ); 
    }
	else if (arg == "-c" || arg == "--config")
	{
		if (i == argc - 1)
			printUsageAndExit(argv[0]);
		config_file = argv[++i];
		lower_case_string(config_file);
	}
	else if (arg == "-o" || arg == "--output")
	{
		if (i == argc - 1)
			printUsageAndExit(argv[0]);
		auto_mode = true;
		task->destination_file = argv[++i];
		lower_case_string(task->destination_file);
	}
	else if (arg == "-sh" || arg == "--shader")
    {
      if (i == argc-1 )
        printUsageAndExit( argv[0] );
      shadername = argv[++i];
      lower_case_string(shadername);
    }
	else if (arg == "--material_override")
	{
		if (i == argc - 1)
			printUsageAndExit(argv[0]);
		material_override_mtl = argv[++i];
		lower_case_string(material_override_mtl);
	}
	else if (arg == "-f" || arg == "--frames")
	{
		auto_mode = true;
		task->destination_samples = stoi(argv[++i]);
	}
	else if (arg == "--rectangle")
	{
		if (i == argc - 1)
			 printUsageAndExit( argv[0] );
		rendering_rect.x = stoi(argv[++i]);
		if (i == argc - 1)
			printUsageAndExit(argv[0]);
		rendering_rect.y = stoi(argv[++i]);
		if (i == argc - 1)
			printUsageAndExit(argv[0]);
		rendering_rect.z = stoi(argv[++i]);
		if (i == argc - 1)
			printUsageAndExit(argv[0]);
		rendering_rect.w = stoi(argv[++i]);
	}
    else 
    {
      filename = argv[i];
      string file_extension;
      size_t idx = filename.find_last_of('.');
      if(idx < filename.length())
      {
        file_extension = filename.substr(idx, filename.length() - idx);
        lower_case_string(file_extension);
      }

      if(file_extension == ".obj")
      {
        filenames.push_back(filename);
        lower_case_string(filenames.back());
      }
    }
  }
  //if ( filenames.size() == 0 ) 
  //  filenames.push_back(string("./meshes/") + "closed_bunny_vn.obj");

  GLUTDisplay::init( argc, argv );

  try 
  {
    ObjScene * scene = new ObjScene( filenames, shadername, config_file, rendering_rect );
	if(material_override_mtl.size() > 0)
		scene->add_override_material_file(material_override_mtl);
	scene->set_render_task(task);
	if (auto_mode)
		scene->start_render_task();
	GLUTDisplay::run( "Optix Renderer", scene );

  } 
  catch( Exception & e )
  {
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }
  return 0;
}
