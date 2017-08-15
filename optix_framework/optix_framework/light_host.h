#include "gui.h"
#include "SampleScene.h"
#include <memory>
#include <optix_world.h>

struct CameraData;

class Light
{
public:
    virtual ~Light() = default;
    std::unique_ptr<CameraData> data;

    Light();
    virtual void set_into_gpu(optix::Context & context);
    virtual void set_into_gui(GUI * gui);

    int get_width() const;
    int get_height() const;
};

