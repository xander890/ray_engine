#include "render_task.h"
#include <float.h>
#include "immediate_gui.h"
#include <sstream>
#include "dialogs.h"
#include <climits>
#include <vector>

RenderTask::RenderTask(std::string destination_file, bool close_program_on_exit) : close_program_on_exit(close_program_on_exit), destination_file(destination_file)
{

}

bool RenderTask::on_draw()
{
	bool changed = false;
	std::vector<char> ss(destination_file.begin(), destination_file.end());
	ImmediateGUIDraw::InputText("Destination file", ss.data(), ImGuiInputTextFlags_ReadOnly);
	if (!is_active() && ImmediateGUIDraw::Button("Choose destination file..."))
	{
		std::string filePath;
		if (Dialogs::saveFileDialog(filePath))
		{
			changed = true;
			destination_file = filePath;
		}
	}

	if(!is_active())
	{
		changed |= ImmediateGUIDraw::Checkbox("Close program on finish", &close_program_on_exit);
	}

	if (is_active())
	{
		std::stringstream ss;
		ss << "Render task in progress. Progress: " << get_progress_string() << std::endl;
		ImmediateGUIDraw::Text(ss.str().c_str());
	}
	return changed;
}

void RenderTask::end()
{
	if (close_program_on_exit)
		exit(2);
}

RenderTaskFrames::RenderTaskFrames(int destination_samples, const std::string& destination_file, bool close_program_on_exit = false ) :
   RenderTask(destination_file, close_program_on_exit), destination_samples(destination_samples), current_frame(INT_MIN)
{
	
}

void RenderTaskFrames::start()
{
	current_frame = 0;
}

 void RenderTaskFrames::update(double /*time*/) {
	 if (is_active())
		 current_frame++; 
 }

bool RenderTaskFrames::is_active()
{
	return current_frame != INT_MIN;
}

bool RenderTaskFrames::is_finished()
{
	return current_frame == destination_samples;
}

 std::string RenderTaskFrames::get_progress_string()
 {
	 return std::to_string(current_frame) + "/" + std::to_string(destination_samples) + " frames";
 }

 void RenderTaskFrames::end() 
{
    RenderTask::end();
	current_frame = INT_MIN;
}

bool RenderTaskFrames::on_draw()
{
	bool changed = false;
	if (!is_active())
	{
		changed |= ImmediateGUIDraw::InputInt("Frames", &destination_samples);
	}
	return changed | RenderTask::on_draw();
}

std::string RenderTaskFrames::to_string()
{
	return std::string("Frame based. Stopping at ") + std::to_string(destination_samples) + " frames";
}

void RenderTaskFrames::update_absolute(double time)
{
	update(time);
}

RenderTaskTime::RenderTaskTime(float destination_time, const std::string &destination_file, bool close_program_on_exit = false):
	RenderTask(destination_file, close_program_on_exit), destination_time(destination_time), current_time(FLT_MIN)
{
}

void RenderTaskTime::start()
{

	current_time = 0.0f;
}

void RenderTaskTime::update(double deltaTime)
{
	if (is_active())
		current_time += deltaTime;
}

void RenderTaskTime::update_absolute(double time)
{
	if (is_active())
	{
		current_time = time;
	}
}

bool RenderTaskTime::is_active()
{
	return current_time != FLT_MIN;
}

bool RenderTaskTime::is_finished()
{
	return current_time > destination_time;
}

std::string RenderTaskTime::get_progress_string()
{
	return std::to_string(current_time) + "/" + std::to_string(destination_time) + " seconds";
}

void RenderTaskTime::end()
{
	RenderTask::end();
	current_time = FLT_MIN;
}

bool RenderTaskTime::on_draw()
{
	bool changed = false;
	if (!is_active())
	{
		changed |= ImmediateGUIDraw::InputFloat("Time (s)", &destination_time);
	}
	return changed | RenderTask::on_draw();
}

std::string RenderTaskTime::to_string()
{
	return std::string("Time based. Stopping at ") + std::to_string(destination_time) + " seconds.";
}



RenderTaskTimeorFrames::RenderTaskTimeorFrames(int max_frames, float dest_time, const std::string & destination_file, bool close_program_on_exit) : RenderTask(destination_file, close_program_on_exit)
{
	destination_samples = max_frames;
	destination_time = dest_time;
	current_time = FLT_MIN;
	current_frame = INT_MIN;
}

void RenderTaskTimeorFrames::start()
{
	current_time = 0.0f;
	current_frame = 0;
}

void RenderTaskTimeorFrames::update(double deltaTime)
{
	if (is_active())
	{
		current_time += deltaTime;
		current_frame += 1;
	}
}

void RenderTaskTimeorFrames::update_absolute(double time)
{
	if (is_active())
	{
		current_time = time;
		current_frame += 1;
	}
}



bool RenderTaskTimeorFrames::is_active()
{
	return current_frame != INT_MIN && current_time != FLT_MIN;
}

bool RenderTaskTimeorFrames::is_finished()
{
	return current_time > destination_time || current_frame >= destination_samples;
}

std::string RenderTaskTimeorFrames::get_progress_string()
{
	return  std::to_string(current_time) + "/" + std::to_string(destination_time) + " seconds,\n" + std::to_string(current_frame) + "/" + std::to_string(destination_samples) + " frames";
}

void RenderTaskTimeorFrames::end()
{
	RenderTask::end();
	current_time = FLT_MIN;
	current_frame = INT_MIN;
}

bool RenderTaskTimeorFrames::on_draw()
{
	bool changed = false;
	if (!is_active())
	{
		changed |= ImmediateGUIDraw::InputInt("Frames", &destination_samples);
		changed |= ImmediateGUIDraw::InputFloat("Time (s)", &destination_time);
	}
	return changed | RenderTask::on_draw();
}

std::string RenderTaskTimeorFrames::to_string()
{
	return std::string("Time/Frame based. Stopping at ") + std::to_string(destination_samples) + " frames or " + std::to_string(destination_time) + " seconds, whichever first.";
}
