#include "image_exporter.h"
#include <fstream>
#include <algorithm>
#include <exception>
#include "IL/il.h"
#include "IL/ilu.h"
#include "logger.h"
#include "texture.h"

inline bool export_raw(const std::string& raw_p, const float * data, int w, int h, int frames)
{
	std::string raw_path = raw_p;
	// export render data
	if (raw_path.length() == 0)
	{
		Logger::error << "Invalid raw file specified" << raw_path << std::endl;
		return false;
	}

	if (raw_path.length() <= 4 || raw_path.substr(raw_path.length() - 4).compare(".raw") != 0)
	{
		raw_path += ".raw";
	}

	std::string txt_file = raw_path.substr(0, raw_path.length() - 4) + ".txt";
	std::ofstream ofs_data(txt_file);
	if (ofs_data.bad())
	{
		Logger::error << "Unable to open file " << txt_file << std::endl;
		return false;
	}
	ofs_data << frames << std::endl << w << " " << h << std::endl;
	ofs_data << 1.0 << " " << 1.0f << " " << 1.0f;
	ofs_data.close();

	RTsize size_buffer = w * h * 4;
	float* mapped = new float[size_buffer];
	memcpy(mapped, data, size_buffer * sizeof(float));
	std::ofstream ofs_image;
	ofs_image.open(raw_path, std::ios::binary);

	if (ofs_image.bad())
	{
		Logger::error << "Error in exporting file" << std::endl;
		return false;
	}

	RTsize size_image = w * h * 3;
	float* converted = new float[size_image];
	float average = 0.0f;
	float maxi = -INFINITY;
	for (int i = 0; i < size_image / 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			if (!isfinite(mapped[i * 4 + j]))
			{
			}
			converted[i * 3 + j] = mapped[i * 4 + j];
			average += mapped[i * 4 + j];
			maxi = fmaxf(maxi, mapped[i * 4 + j]);
		}
	}
	average /= size_image * 3;
	delete[] mapped;
	ofs_image.write(reinterpret_cast<const char*>(converted), size_image * sizeof(float));
	ofs_image.close();
	delete[] converted;
	Logger::info << "Exported buffer to " << raw_path << " (avg: " << std::to_string(average) << ", max: " << std::to_string(maxi) << ")" << std::endl;

	return true;
}

bool export_devil(const std::string& raw_p, const float * data, int w, int h)
{
	ILuint	imgId;
	ilInit();
	ilGenImages(1, &imgId);
	ilBindImage(imgId);
	ilSetInteger(IL_JPG_QUALITY, 99);
	ilSetInteger(IL_PNG_INTERLACE, IL_FALSE);

	int s = w * h * 4;
	std::vector<unsigned char> data_cast(s, 0);
	for (int i = 0; i < data_cast.size(); i++)
	{
		data_cast[i] = static_cast<unsigned char>(optix::clamp(data[i], 0.0f, 1.0f) * 255.0f);
	}
	ilTexImage(w, h, 1, 4, IL_RGBA, IL_UNSIGNED_BYTE, &data_cast[0]);

	ilEnable(IL_FILE_OVERWRITE);
	auto a = ilGetError();
	ilSaveImage(raw_p.c_str());
	ilDeleteImages(1, &imgId);
	ilShutDown();
	return true;
}

bool exportTexture(const std::string & filename, const float * data, int w, int h, int past_frames)
{
	std::string filename_lc = filename;
	std::transform(filename.begin(), filename.end(), filename_lc.begin(), ::tolower);
	size_t len = filename.length();
	bool success = false;

	if (len >= 4)
	{
		std::string ext = filename_lc.substr(filename_lc.length() - 4);
		if (ext.compare(".raw") == 0)
		{
			success = export_raw(filename, data, w, h, past_frames);
		}
		else
		{
			success = export_devil(filename, data, w, h);
		}
	}
	return success;
}

bool exportTexture(const std::string & filename, const std::unique_ptr<Texture>& tex, const int past_frames)
{
	assert(tex->get_dimensionality() == 2);
	return exportTexture(filename, (float*)tex->get_data(), tex->get_width(), tex->get_height(), past_frames);
}

bool exportTexture(const std::string & filename, optix::Buffer buf, const int past_frames)
{
	assert(buf->getDimensionality() == 2);
	void *data = buf->map();
	size_t w, h;
	buf->getSize(w, h);
	bool res = exportTexture(filename, (float*)data, w, h, past_frames);
	buf->unmap();
	return res;
}
