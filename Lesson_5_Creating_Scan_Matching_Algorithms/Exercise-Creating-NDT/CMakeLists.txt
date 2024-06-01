project(Localization)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_minimum_required (VERSION 3.10)

# -g allows for gdb debugging
# turn on -03 for best performance
add_definitions(-std=c++14 -O3)

set(CXX_FLAGS "-pthread -fPIC -I..")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(sources ndt-main.cpp helper.cpp)
set(sources_sol solution/ndt-main-solution.cpp helper.cpp)

find_package(PCL 1.4 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

add_executable(ndt ${sources} )
add_executable(ndt_solution ${sources_sol} )

target_link_libraries(ndt ${PCL_LIBRARIES})
target_link_libraries(ndt_solution ${PCL_LIBRARIES})
