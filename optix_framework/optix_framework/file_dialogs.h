#pragma once
#include <string>

class Dialogs
{
public:
	static bool openFileDialog(std::string & selectedFilePath, const std::string & filters = "");
	static bool saveFileDialog(std::string & selectedFilePath, const std::string & filters = "");
};
