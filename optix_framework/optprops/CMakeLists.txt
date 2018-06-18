set(HEADERS}
cdf_bsearch.h
glass.h
incandescent.h
Interface.h
load_mpml.h
LorenzMie.h
Medium.h
milk.h
mt_random.h
Randomizer.h
spectrum2rgb.h
spectrum2xyz.h
water.h
string_utils.h
XmlParser.h
Parse.h
minimal_algembra.h
)

set(SRCS    glass.cpp
            incandescent.cpp
            load_mpml.cpp
            LorenzMie.cpp
            Medium.cpp
            milk.cpp
            Randomizer.cpp
            string_utils.cpp
            XmlParser.cpp
            water.cpp
			Parse.cpp
    )

source_group("Headers" FILES ${HEADERS})
source_group("Source Files" FILES ${SRCS})
    
add_library( optprops STATIC
             ${SRCS} ${HEADERS} )
target_include_directories( optprops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)             
target_link_libraries(optprops)