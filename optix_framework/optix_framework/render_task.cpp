#include "render_task.h"
#include <float.h>
#include "immediate_gui.h"
#include <sstream>

RenderTask::RenderTask(std::string destination_file, bool close_program_on_exit) : close_program_on_exit(close_program_on_exit), destination_file(destination_file)
{

}

bool RenderTask::on_draw()
{
	bool changed = false;
	char InputBuf[256];
	sprintf_s(InputBuf, "%s", get_destination_file().c_str());
	if (!is_active() && ImmediateGUIDraw::InputText("Destination file", InputBuf, ImGuiInputTextFlags_EnterReturnsTrue))
	{
		changed = true;
		destination_file = std::string(InputBuf);
	}

	if(!is_active())
	{
		changed |= ImmediateGUIDraw::Checkbox("Close program on finish", &close_program_on_exit);
	}
	if (!is_active() && ImmediateGUIDraw::Button("Start task"))
	{
		changed = true;
		start();
	}

	if (is_active())
	{
		std::stringstream ss;
		ss << "Render task in progress. Progress: " << get_progress_string() << std::endl;
		ImmediateGUIDraw::Text(ss.str().c_str());
	}
	if (is_active() && ImmediateGUIDraw::Button("End task"))
	{
		end();
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

 void RenderTaskFrames::update(float /*time*/) { 
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

RenderTaskTime::RenderTaskTime(float destination_time, const std::string& destination_file, bool close_program_on_exit = false):
	RenderTask(destination_file, close_program_on_exit), destination_time(destination_time), current_time(FLT_MIN)
{
}

void RenderTaskTime::start()
{

	current_time = 0.0f;
}

void RenderTaskTime::update(float deltaTime)
{
	if (is_active())
		current_time += deltaTime;
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