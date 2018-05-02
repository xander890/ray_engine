#ifndef  __FOLDERS_H__
#define  __FOLDERS_H__
#include <string>

class Folders
{
public:
	static void init();	
	static std::string mpml_file;
	static std::string data_folder;
	static std::string merl_folder;
	static std::string merl_database_file;
	static std::string ptx_path;
};

static const char* const get_path_ptx(const std::string& base)
{
	static std::string path;
	path = Folders::ptx_path + "/framework_generated_" + base + ".ptx";
	return path.c_str();
}

#endif /* __FOLDERS_H__ */


