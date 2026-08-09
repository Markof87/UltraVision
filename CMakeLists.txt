cmake_minimum_required(VERSION 3.15)
project(UltraVision)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

file(GLOB SOURCES "*.cpp")
add_executable(UltraVision ${SOURCES})
target_link_libraries(UltraVision PRIVATE ${OpenCV_LIBS})