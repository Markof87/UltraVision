#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <map>

using namespace cv;

int main(int argc, char **argv)
{
    const std::string BASE_PATH = "../Sequences";

    std::string path;
    std::string windowName; 

    //Load all image files from the specified path and sort them in ascending order
    std::map<std::string, std::vector<std::string>> imageMap;

    if(argc == 4)
    {
        path = BASE_PATH + "/" + argv[1] + "/person0" + argv[2] + "_" + argv[1] + "_" + argv[3] + "/data";
        windowName = "Person 0" + std::string(argv[2]) + " " + std::string(argv[1]) + " scenario d" + std::string(argv[3]);
        
        for (const auto &entry : std::filesystem::directory_iterator(path))
            imageMap[windowName].push_back(entry.path().string());
    }

    if(argc == 3) 
    {
        path = BASE_PATH + "/" + argv[1] + "/person0" + argv[2] + "_" + argv[1];

        for (int i = 1; i <= 4; i++) {
            std::string subfolderPath = path + "_d" + std::to_string(i) + "/data";
            windowName = "Person 0" + std::string(argv[2]) + " " + std::string(argv[1]) + " scenario d" + std::to_string(i);
            for (const auto &entry : std::filesystem::directory_iterator(subfolderPath))
                imageMap[windowName].push_back(entry.path().string());
        }
    }

    if(argc == 2) 
    {
        path = BASE_PATH + "/" + argv[1];
        for (int i = 1; i <=3; i++)
        {
            for (int j = 1; j <= 4; j++) 
            {
                std::string subfolderPath = path + "/person0" + std::to_string(i) + "_" + argv[1] + "_d" + std::to_string(j) + "/data";
                windowName = "Person 0" + std::to_string(i) + " " + std::string(argv[1]) + " scenario d" + std::to_string(j);
                for (const auto &entry : std::filesystem::directory_iterator(subfolderPath))
                    imageMap[windowName].push_back(entry.path().string());
            }
        }
    }

    if(argc == 1) 
    {
        path = BASE_PATH;
        std::string actions[] = {"boxing", "handclapping", "handwaving", "jogging", "running", "walking"};
        for (const auto &action : actions) 
        {
            for (int i = 1; i <= 3; i++) 
            {
                for (int j = 1; j <= 4; j++) 
                {
                    std::string subfolderPath = path + "/" + action + "/person0" + std::to_string(i) + "_" + action + "_d" + std::to_string(j) + "/data";
                    windowName = "Person 0" + std::to_string(i) + " " + action + " scenario d" + std::to_string(j);

                    for (const auto &entry : std::filesystem::directory_iterator(subfolderPath))
                        imageMap[windowName].push_back(entry.path().string());
                }
            }
        }
    }
    
    //Loop all the images and display them in a window like a sequence
    for (const auto &pair : imageMap)
    {
        const std::string &windowName = pair.first;
        const std::vector<std::string> &files = pair.second;

        for (const auto &imageFile : files)
        {
            Mat img = imread(imageFile);

            // Check if the image exists and can be opened
            if (img.empty())
            {
                std::cerr << "Could not open or find the image: " << imageFile << std::endl;
                continue; // Skip to the next image
            }

            // Open window and show the image
            namedWindow(windowName);
            imshow(windowName, img);

            // Wait for 40ms (25 FPS) before showing the next image
            waitKey(40);
            
        }
    }

    char k = waitKey(0);

    return 0;
}