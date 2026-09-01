/**
 * @file Sequencer.cpp
 * @brief Implementation of the Sequencer class, which handles the sequencing of images.
 * @author Marco Fabiani
 */

#include "Sequencer.h"
#include "ManipulatingService.h"
#include "utils/Utils.h"

void Sequencer::run(const std::string& basePath, const std::string& windowName)
{
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    cv::createTrackbar("Action", windowName, nullptr, 5, [](int pos, void* target){
        static_cast<Sequencer*>(target)->setCurrentActionIdx(pos);
    }, this);

    cv::createTrackbar("Person", windowName, nullptr, 2, [](int pos, void* target){
        static_cast<Sequencer*>(target)->setCurrentPersonIdx(pos);
    }, this);

    cv::createTrackbar("Scenario", windowName, nullptr, 3, [](int pos, void* target){
        static_cast<Sequencer*>(target)->setCurrentScenarioIdx(pos);
    }, this);

    std::vector<cv::Mat> loadedSequence;
    cv::Mat canvas;

    while(true)
    {
        updateLoader(loadedSequence, basePath);

        if(!loadedSequence.empty())
        {
            cv::Mat currentRawFrame = loadedSequence[frame_idx];

            cv::Mat segmentedMask;
            ManipulatingService::manipulateFrame(currentRawFrame, segmentedMask);

            drawHUD(canvas, segmentedMask, loadedSequence.size());
            cv::imshow(windowName, canvas);

            if(play_state == 1)
            {
                frame_idx = (frame_idx + 1) % loadedSequence.size();
                cv::waitKey(40); // Wait for 40ms (25 FPS)
            }
        }

        int delay = (play_state == 1) ? 40 : 100; // 40ms for playing, 100ms for paused
        char key = static_cast<char>(cv::waitKey(delay));

        if(handleKeyboard(key, loadedSequence.size()))
            break;
    }

    cv::destroyWindow(windowName);
}

void Sequencer::updateLoader(std::vector<cv::Mat>& loadedSequence, const std::string& basePath)
{
    const std::string actions[] = {"boxing", "handclapping", "handwaving", "jogging", "running", "walking"};

    if(need_reload)
    {
        std::string selected_action = actions[current_action_idx];
        int selected_person = current_person_idx + 1;
        int selected_scenario = current_scenario_idx + 1;

        std::cout << "Loading dataset for action: " << selected_action 
                  << ", person: " << selected_person 
                  << ", scenario: " << selected_scenario << std::endl;

        // Load the dataset based on the selected parameters
        loadedSequence = Utils::loadDataset(basePath, selected_action, selected_person, selected_scenario);
        frame_idx = 0;
        need_reload = false;
    }
}

void Sequencer::drawHUD(cv::Mat& canvas, const cv::Mat& currentFrame, size_t totalFrames)
{
    //Interactive HUD
    canvas = cv::Mat::zeros(600, 800, CV_8UC3);
    cv::Mat frameResized;
    cv::resize(currentFrame, frameResized, cv::Size(800, 600));
    frameResized.copyTo(canvas(cv::Rect(0, 0, frameResized.cols, frameResized.rows)));

    std::string statusText = "STATUS: " + std::string(play_state == 1 ? "PLAYING" : "PAUSED");
    cv::putText(canvas, statusText, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    std::string frameText = "FRAME: " + std::to_string(frame_idx + 1) + "/" + std::to_string(totalFrames);
    cv::putText(canvas, frameText, cv::Point(340, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

    cv::line(canvas, cv::Point(0, 50), cv::Point(800, 50), cv::Scalar(0, 255, 0), 1);

    //Put a legend for the keyboard controls
    std::string controlsText = "Controls: [p] Play/Pause | [m] Next Frame | [n] Previous Frame | [q] Quit";
    cv::putText(canvas, controlsText, cv::Point(10, 580), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
}

bool Sequencer::handleKeyboard(char key, size_t totalFrames)
{
    if(key == 'q' || key == 27) // 'q' or ESC to quit
        return true;
    else if(key == 'p') // 'p' to toggle play/pause
        play_state = 1 - play_state; // Toggle between 0 and 1
    else if(key == 'm') // 'm' for next frame
    {
        if(play_state == 0 && totalFrames > 0)
            frame_idx = (frame_idx + 1) % totalFrames;
    }
    else if(key == 'n') // 'n' for previous frame
    {
        if(play_state == 0 && totalFrames > 0)
            frame_idx = (frame_idx - 1 + totalFrames) % totalFrames;
    }

    return false;
}