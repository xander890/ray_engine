
#include "SampleScene.h"
#include <memory>
#include <optix_world.h>

struct LightData;

class Light
{
public:
    virtual ~Light() = default;
    std::unique_ptr<LightData> data;

    Light();

    int get_width() const;
    int get_height() const;
};

