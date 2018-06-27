#pragma once
#include <string>

/*
 * Interface to the NFD library, used to create system independent open and save file dialogs.
 */
class Dialogs
{
public:
	static bool open_file_dialog(std::string & selectedFilePath, const std::string & filters = "");
	static bool save_file_dialog(std::string & selectedFilePath, const std::string & filters = "");
};
