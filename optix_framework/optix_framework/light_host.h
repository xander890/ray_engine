
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

    int get_width() const;
    int get_height() const;
};

