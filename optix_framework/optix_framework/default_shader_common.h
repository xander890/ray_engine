#pragma once

#include "host_device_common.h"

// TODO implement me. 
// When visualizing an interface, implement a way to debug so that only some effects are shown.
#define IMPROVED_ENUM_NAME InterfaceVisualizationFlags
#define IMPROVED_ENUM_LIST ENUMITEM_VALUE(VISUALIZE_REFLECTION,0x01) \
                           ENUMITEM_VALUE(VISUALIZE_REFRACTION,0x02) \
						   ENUMITEM_VALUE(VISUALIZE_ABSORPTION,0x04)
#include "improved_enum.inc"
