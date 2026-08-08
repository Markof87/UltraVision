# 1. Diciamo quale versione di CMake usare 
cmake_minimum_required(VERSION 3.10)

# 2. Diamo un nome al nostro progetto
project(ActionRecognitionProject)
set(CMAKE_CXX_STANDARD 17) #importante dato che usiamo <filesystem>

# 3. Diciamo a CMake di cercare la libreria OpenCV nel computer
find_package(OpenCV REQUIRED)

# 4. Diciamo a CMake di creare il nostro programma partendo da main.cpp
add_executable(ActionRecognitionProject main.cpp utils.hpp)

# 5. Uniamo il nostro programma alle librerie di OpenCV appena trovate
target_link_libraries(ActionRecognitionProject ${OpenCV_LIBS})