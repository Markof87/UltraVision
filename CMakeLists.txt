cmake_minimum_required(VERSION 3.15)
project(UltraVision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(include)

add_library(UltraVisionCore STATIC
    src/FilterTechniques.cpp
    src/SegmentationTechniques.cpp
    src/utils/Utils.cpp
    src/TestingFunctions.cpp
    src/TrackingPerson.cpp
)

target_link_libraries(UltraVisionCore PRIVATE ${OpenCV_LIBS})

add_executable(UltraVision src/main.cpp)
target_link_libraries(UltraVision PRIVATE UltraVisionCore ${OpenCV_LIBS})

#Comment this line if you want to build in command mode
#add_subdirectory(gui)