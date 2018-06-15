#include "dialogs.h"
#include "nfd/include/nfd.h"


bool Dialogs::openFileDialog(std::string& selectedFilePath, const std::string& filters)
{
	nfdchar_t *outPath = nullptr;
	nfdresult_t result = NFD_OpenDialog(filters.c_str(), nullptr, &outPath);
	if (result == NFD_ERROR)
	{
		puts(NFD_GetError());
		return false;
	}

	if (result == NFD_OKAY)
	{ 
		selectedFilePath = std::string(outPath);
		return true;
	}
	selectedFilePath = std::string("");
	return false;
}

bool Dialogs::saveFileDialog(std::string& selectedFilePath, const std::string& filters)
{
	nfdchar_t *outPath = nullptr;
	nfdresult_t result = NFD_SaveDialog(filters.c_str(), nullptr, &outPath);
	if (result == NFD_OKAY)
	{
		selectedFilePath = std::string(outPath);
		return true;
	}
	selectedFilePath = std::string("");
	return false;
}
