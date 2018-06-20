#include <device_common.h>
#include <color_utils.h>
#include <ray_tracing_utils.h>
#include <environment_map.h>

using namespace optix;

// Window variables
rtBuffer<uchar4, 2> tonemap_output_buffer;
rtBuffer<uchar4, 2> debug_output_buffer;

rtDeclareVariable(optix::uint4, zoom_window, , );
rtDeclareVariable(optix::uint4, image_part_to_zoom, , );

_fn bool box(const uint2& px, const uint4& bx)
{
	return px.x >= bx.x && px.x < bx.x + bx.z && px.y >= bx.y && px.y < bx.y + bx.w;
}

_fn bool box_border(const uint2& px, const uint4& bx)
{
	return box(px, make_uint4(bx.x - 1, bx.y - 1, bx.z + 2, bx.w + 2)) && !box(px, bx);
}


RT_PROGRAM void debug_camera()
{
	uint2 current_pixel = launch_index;

	if (box(current_pixel, zoom_window))
	{
		uint2 source_pixel;
		source_pixel.x = uint(((current_pixel.x - zoom_window.x) / (float)zoom_window.z * image_part_to_zoom.z) + image_part_to_zoom.x);
		source_pixel.y = uint(((current_pixel.y - zoom_window.y) / (float)zoom_window.w * image_part_to_zoom.w) + image_part_to_zoom.y);
		debug_output_buffer[launch_index] = tonemap_output_buffer[source_pixel];

		uint4 debug_pixel_bbox;
		debug_pixel_bbox.x = uint((debug_index.x - image_part_to_zoom.x) / (float)image_part_to_zoom.z * zoom_window.z + zoom_window.x);
		debug_pixel_bbox.y = uint((debug_index.y - image_part_to_zoom.y) / (float)image_part_to_zoom.w * zoom_window.w + zoom_window.y);
		debug_pixel_bbox.z = uint(zoom_window.z / (float)image_part_to_zoom.z);
		debug_pixel_bbox.w = uint(zoom_window.w / (float)image_part_to_zoom.w);

		if (box_border(current_pixel, debug_pixel_bbox))
		{
			debug_output_buffer[launch_index] = make_uchar4(0, 0, 255, 0);
		}
	}
	else
	{
		if(box_border(current_pixel, image_part_to_zoom))
			debug_output_buffer[launch_index] = make_uchar4(0, 0, 255, 0);
		else
			debug_output_buffer[launch_index] = tonemap_output_buffer[launch_index];
	}
}
