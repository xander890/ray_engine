#pragma once
#include <string>

class RenderTask
{
public:
	bool close_program_on_exit = false;
	int destination_samples = 1000;
	std::string destination_file = "./result.raw";
	void start();
	void update();
	bool is_active();
	bool is_finished();
	int get_progress();
	void end();
private:
	int current_frame = -1;
};