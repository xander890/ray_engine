set(INCLUDE_DIR include)
include_directories(${INCLUDE_DIR})

set(HEADERS
common.h
nfd_common.h
${INCLUDE_DIR}/nfd.h
)

set(SRCS    nfd_common.c)

source_group("Headers" FILES ${HEADERS})
source_group("Source Files" FILES ${SRCS})

if (WIN32)
ADD_LIBRARY( nfd STATIC ${SRCS} ${HEADERS} nfd_win.cpp )
endif (WIN32)

if (UNIX)
ADD_LIBRARY( nfd STATIC ${SRCS} ${HEADERS} nfd_gtk.c )
target_link_libraries(nfd ${GTK3_LIBRARIES})
endif (UNIX)
