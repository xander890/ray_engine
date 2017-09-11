#ifndef  __FOLDERS_H__
#define  __FOLDERS_H__
#include <string>
#include "parameter_parser.h"
#define PATH_TO_MY_PTX_FILES  "PTX_files" 


class Folders
{
public:
	static void init();	
	static std::string mpml_file;
	static std::string data_folder;
	static std::string merl_folder;
	static std::string merl_database_file;
	static std::string texture_folder;
	static std::string ptx_path;
};

static const char* const get_path_ptx(const std::string& base)
{
	static std::string path;
	path = Folders::ptx_path + "/" + base + ".ptx";
	return path.c_str();
}

#define DEFAULT_TEXTURE_FOLDER std::string("./images/")


#endif /* __FOLDERS_H__ */


