#include "folders.h"
#include "logger.h"

std::string Folders::mpml_file = std::string("");
std::string Folders::merl_folder = std::string("");
std::string Folders::data_folder = std::string("");
std::string Folders::merl_database_file = std::string("");
std::string Folders::texture_folder = std::string("");
std::string Folders::shader_folder = std::string("");
std::string Folders::ptx_path = std::string("");

void Folders::init()
{
	data_folder = ParameterParser::get_parameter("folders","data_folder", std::string("./"), "Folder with all the data necessary for rendering excluding configuration files. All model/file paths will be relative to this folder.");
	mpml_file = data_folder + ParameterParser::get_parameter("folders","mpml_file", std::string("/mpml/media.mpml"), "MPML File for material properties.");
	merl_folder = data_folder + ParameterParser::get_parameter("folders","merl_folder", std::string("brdf/"), "Folder where to look for MERL database BRDFs.");
	merl_database_file = merl_folder + ParameterParser::get_parameter("folders","merl_database", std::string("database.txt"), "Text file that contains a list of the MERL data to load.");
	texture_folder = data_folder + ParameterParser::get_parameter("folders","texture_folder", std::string("images/"), "Image folder for MTL files.");
	shader_folder = data_folder + ParameterParser::get_parameter("folders","shader_folder", std::string("shaders/"), "Shader folder.");
	ptx_path = ParameterParser::get_parameter("folders","ptx_path", std::string("PTX_files"), "Compiled cuda files folder. Visual studio needs to be also set in case to output the files to the right folder!");
	Logger::info << "Folders: \n" <<
		"Data: " << data_folder << "\n" <<
		"MPML: " << mpml_file << "\n" <<
		"MERL: " << merl_folder << "\n" <<
		"MERLdb: " << merl_database_file << "\n" <<
		"Textures: " << texture_folder << "\n" <<
		"Shaders: " << shader_folder << "\n" <<
		"PTX: " << ptx_path << "\n" << endl;
}

