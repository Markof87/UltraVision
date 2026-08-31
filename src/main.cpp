#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>

#include "utils/Utils.h"
#include "TestingFunctions.h"
#include "TrackingPerson.h"

using namespace cv;

int main(int argc, char **argv)
{
    TrackingPerson tracker = TrackingPerson();

    const std::string BASE_PATH = "../data/Sequences";
    const std::string WINDOW_NAME = "UltraVision Sequence Viewer";

    // Command line parser to handle input arguments
    const std::string keys="{path |../data/Sequences | Path to the dataset folder}"
                             "{action | | Action name (boxing, handclapping, handwaving, jogging, running, walking)}"
                             "{person | -1 | Person ID (1, 2, 3)}"
                             "{scenario | -1 | Scenario ID (1, 2, 3, 4)}";
    cv::CommandLineParser parser(argc, argv, keys);

    std::string action;
    int personId;
    int scenarioId;

    // Load command line argument parameters
    Utils::loadArgumentParameters(parser, action, personId, scenarioId);

    // Load the dataset based on the provided parameters
    std::vector<cv::Mat> imageMap = Utils::loadDataset(BASE_PATH, action, personId, scenarioId);
    
    std::vector<cv::Mat> otsuResults;
    std::vector<cv::Rect> boundingBoxes;
    // Test the otsu four regions function
    for(const auto& img : imageMap)
    { 
        //Clone original image with grayscale
        cv::Mat greyImg, otsuImg;
        cv::cvtColor(img, greyImg, cv::COLOR_BGR2GRAY);

        otsuImg = TestingFunctions::test_otsu_four_regions(greyImg);
        otsuResults.push_back(otsuImg);

        boundingBoxes.push_back(tracker.getActorBoundingBox(otsuImg));
    }
    
    Utils::showDataset(otsuResults, boundingBoxes, WINDOW_NAME);

    return 0;
}