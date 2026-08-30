#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>

#include "utils/Utils.h"
#include "TestingFunctions.h"

using namespace cv;

int main(int argc, char **argv)
{
    const std::string BASE_PATH = "../data/Sequences";
    const std::string WINDOW_NAME = "UltraVision Sequence Viewer";
    const std::string actions[] = {"boxing", "handclapping", "handwaving", "jogging", "running", "walking"};

    //Load all image files from the specified path and sort them in ascending order
    std::vector<cv::Mat> imageMap;

    // Command line parser to handle input arguments
    const std::string keys="{path |../data/Sequences | Path to the dataset folder}"
                             "{action | | Action name (boxing, handclapping, handwaving, jogging, running, walking)}"
                             "{person | -1 | Person ID (1, 2, 3)}"
                             "{scenario | -1 | Scenario ID (1, 2, 3, 4)}";
    cv::CommandLineParser parser(argc, argv, keys);

    std::string action = parser.get<std::string>("action");
    int personId = parser.get<int>("person");
    int scenarioId = parser.get<int>("scenario");

    // Check if the command line arguments are valid
    if(!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    // if the lowercase of action is not in the list of actions, print an error message and exit
    if(!action.empty() && std::find(std::begin(actions), std::end(actions), action) == std::end(actions))
    {
        std::cerr << "Invalid action name: " << action << std::endl;
        return -1;
    }

    // if personId is not 1, 2, or 3, print an error message and exit
    if(personId != 0 && (personId < 1 || personId > 3))
    {
        std::cerr << "Invalid person ID: " << personId << std::endl;
        return -1;
    }

    // if scenarioId is not 1, 2, 3, or 4, print an error message and exit
    if(scenarioId != 0 && (scenarioId < 1 || scenarioId > 4))
    {
        std::cerr << "Invalid scenario ID: " << scenarioId << std::endl;
        return -1;
    }

    // Load the dataset based on the provided parameters
    imageMap = Utils::loadDataset(BASE_PATH, action, personId, scenarioId);
    
    std::vector<cv::Mat> otsuResults;
    // Test the otsu four regions function
    for(const auto& img : imageMap)
    { 
        //Clone original image with grayscale
        cv::Mat greyImg;
        cv::cvtColor(img, greyImg, cv::COLOR_BGR2GRAY);

        //cv::Mat greyImg = img.clone(cv::IMREAD_GRAYSCALE);
        otsuResults.push_back(TestingFunctions::test_otsu_four_regions(greyImg));
    }
    
    Utils::showDataset(otsuResults, WINDOW_NAME);

    return 0;
}