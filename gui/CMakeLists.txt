cmake_minimum_required(VERSION 3.15)
project(UltraVisionGUI)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(../include)

set(GUI_SOURCES
    main_gui.cpp
    MainWindow.cpp
    MainWindow.ui
)

add_executable(UltraVisionGUI ${GUI_SOURCES})
target_link_libraries(UltraVisionGUI PRIVATE UltraVisionCore Qt5::Widgets ${OpenCV_LIBS})