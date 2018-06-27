#pragma once
#include <string>
#include "optix_serialize_utils.h"
#include <fstream>

// Check if a file exists
inline bool exists(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

/*
* Utility class for paths. Generically, the renderer goes to look some files into a "data" folder, that can be user speficied in a configuration file. TODO this will be progressively removed in later versions thanks to the new gui and serializtion, but for now we have to keep it like this.
*
* ptx_path is a generic folder in which we store the compiled ptxs.
*/
class Folders
{
public:
	static std::string data_folder;
	static std::string ptx_path;

    static std::string get_path_to_ptx(const std::string& base)
    {
        static std::string path;
        path = Folders::ptx_path + "/framework_generated_" + base + ".ptx";
        return path;
    }
};





