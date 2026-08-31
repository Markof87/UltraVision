#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>

#include "utils/Utils.h"
#include "TestingFunctions.h"
#include "TrackingPerson.h"

const std::string BASE_PATH = "../data/Sequences";
const std::string WINDOW_NAME = "UltraVision Sequence Viewer";
const std::string actions[] = {"boxing", "handclapping", "handwaving", "jogging", "running", "walking"};

//Global status interface
int current_action_idx = 0;
int current_person_idx = 0;
int current_scenario_idx = 0;
int play_state = 0; // 0: paused, 1: playing

bool need_reload = true;

// Callback function for trackbar changes
void onTrackbarChange(int, void*)
{
    need_reload = true;
}

int main(int argc, char **argv)
{
    std::vector<cv::Mat> loadedSequence;

    // If input arguments are empty, use graphical interface
    if(argc == 1)
    {
        cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

        //From 0 to 5 for actions, 1 to 3 for persons, 1 to 4 for scenarios
        cv::createTrackbar("Action", WINDOW_NAME, &current_action_idx, 5, onTrackbarChange);
        cv::createTrackbar("Person", WINDOW_NAME, &current_person_idx, 2, onTrackbarChange);
        cv::createTrackbar("Scenario", WINDOW_NAME, &current_scenario_idx, 3, onTrackbarChange);

        int frame_idx = 0;

        while(true)
        {
            if(need_reload)
            {
                std::string selected_action = actions[current_action_idx];
                int selected_person = current_person_idx + 1;
                int selected_scenario = current_scenario_idx + 1;
                
                // Load the dataset based on the selected parameters
                loadedSequence = Utils::loadDataset(BASE_PATH, selected_action, selected_person, selected_scenario);
                frame_idx = 0;
                need_reload = false;
            }

            if(!loadedSequence.empty())
            {
                cv::Mat frameResized;
                cv::resize(loadedSequence[frame_idx], frameResized, cv::Size(800, 600));

                //Interactive HUD
                cv::Mat canvas = cv::Mat::zeros(600, 800, CV_8UC3);
                frameResized.copyTo(canvas(cv::Rect(0, 0, frameResized.cols, frameResized.rows)));

                std::string statusText = "STATUS: " + std::string(play_state == 1 ? "PLAYING" : "PAUSED");
                cv::putText(canvas, statusText, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

                std::string frameText = "FRAME: " + std::to_string(frame_idx + 1) + "/" + std::to_string(loadedSequence.size());
                cv::putText(canvas, frameText, cv::Point(340, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

                cv::line(canvas, cv::Point(0, 50), cv::Point(800, 50), cv::Scalar(0, 255, 0), 1);

                cv::imshow(WINDOW_NAME, canvas);
                if(play_state == 1)
                {
                    frame_idx = (frame_idx + 1) % loadedSequence.size();
                    cv::waitKey(40); // Wait for 40ms (25 FPS)
                }
            }
            int delay = (play_state == 1) ? 40 : 100; // 40ms for playing, 100ms for paused
            char key = static_cast<char>(cv::waitKey(delay));

            if(key == 'q' || key == 27) // 'q' or ESC to quit
                break;
            else if(key == 'p') // 'p' to toggle play/pause
                play_state = 1 - play_state; // Toggle between 0 and 1
            else if(key == 'n') // 'n' for next frame
            {
                if(play_state == 0 && !loadedSequence.empty())
                    frame_idx = (frame_idx + 1) % loadedSequence.size();
            }
            else if(key == 'b') // 'b' for previous frame
            {
                if(play_state == 0 && !loadedSequence.empty())
                    frame_idx = (frame_idx - 1 + loadedSequence.size()) % loadedSequence.size();
            }
        }
    }

    else
    {
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

        // Load the dataset based on the provided command line parameters
        loadedSequence = Utils::loadDataset(BASE_PATH, action, personId, scenarioId);

        if(loadedSequence.empty())
        {
            std::cerr << "No images found for the specified parameters." << std::endl;
            return -1;
        }

        // Display the loaded sequence
        for(const auto& img : loadedSequence)
        {
            cv::Mat frameResized;
            cv::resize(img, frameResized, cv::Size(800, 600));
            cv::imshow(WINDOW_NAME, frameResized);
            char key = static_cast<char>(cv::waitKey(40)); // Wait for 40ms (25 FPS)
            if(key == 'q' || key == 27) // 'q' or ESC to quit
                break;
        }
    }


    // Load the dataset based on the provided parameters
    //std::vector<cv::Mat> imageMap = Utils::loadDataset(BASE_PATH, action, personId, scenarioId);
    
    std::vector<cv::Mat> otsuResults;
    std::vector<cv::Rect> boundingBoxes;


    // Test the otsu four regions function
    /*for(const auto& img : imageMap)
    { 
        //Clone original image with grayscale
        cv::Mat greyImg, otsuImg;
        cv::cvtColor(img, greyImg, cv::COLOR_BGR2GRAY);

        otsuImg = TestingFunctions::test_otsu_four_regions(greyImg);
        otsuResults.push_back(otsuImg);

        // Get the bounding box of the actor in the image
        boundingBoxes.push_back(tracker.getActorBoundingBox(otsuImg));
    }*/
    
    //Utils::showDataset(otsuResults, boundingBoxes, WINDOW_NAME);

    return 0;
}