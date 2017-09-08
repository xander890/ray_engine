#include "render_task.h"

 void RenderTask::start() { current_frame = 0; }

 void RenderTask::update() { 
	 if(current_frame != -1)
		 current_frame++; 
 }

bool RenderTask::is_active()
{
	return current_frame != -1;
}

 bool RenderTask::is_finished() { return current_frame == destination_samples; }

int RenderTask::get_progress()
{
	return current_frame;
}

 void RenderTask::end() 
{
	if (close_program_on_exit)
		exit(2);
	current_frame = -1;
}
