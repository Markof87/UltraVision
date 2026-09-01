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
    src/Sequencer.cpp
    src/ManipulatingService.cpp
    src/TestingFunctions.cpp

    # questi sotto servono per avviare main_segmentation
    tecniche_segmentazione.h
    segm_files/otsu_watershed_meanshift.cpp
    segm_files/otsu_then_watershed.cpp
    segm_files/otsu_4regions.cpp
    segm_files/otsu_2regions.cpp
    segm_files/sobel_then_otsu.cpp
    segm_files/stretching_then_sobel_then_otsu.cpp
    segm_files/otsu_2reg_dynamic.cpp
    segm_files/watershed.cpp
    segm_files/prova_watershed.cpp
    segm_files/prova2_watershed.cpp

    
    src/TrackingPerson.cpp
)

target_link_libraries(UltraVisionCore PRIVATE ${OpenCV_LIBS})

add_executable(UltraVision #main_segmentation.cpp
                            src/main.cpp
                            )
target_link_libraries(UltraVision PRIVATE UltraVisionCore ${OpenCV_LIBS})

#Comment this line if you want to build in command mode
#add_subdirectory(gui)