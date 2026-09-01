#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>

#include "utils/Utils.h"
#include "Sequencer.h"
#include "TestingFunctions.h"
#include "TrackingPerson.h"

const std::string BASE_PATH = "../data/Sequences";
const std::string WINDOW_NAME = "UltraVision Sequence Viewer";

int main(int argc, char **argv)
{
    Sequencer sequencer = Sequencer();
    std::vector<cv::Mat> loadedSequence;

    // If input arguments are empty, use graphical interface
    if(argc == 1)
    {
        sequencer.run(BASE_PATH, WINDOW_NAME);
    }

    else
    {
        // Command line parser to handle input arguments
        const std::string keys="{path |../data/Sequences | Path to the dataset folder}"
                                "{action | | Action name (boxing, handclapping, handwaving, jogging, running, walking)}"
                                "{person | -1 | Person ID (1, 2, 3)}"
                                "{scenario | -1 | Scenario ID (1, 2, 3, 4)}";
        cv::CommandLineParser parser(argc, argv, keys);

        Utils::loadSequenceFromCommandLineArguments(loadedSequence, BASE_PATH, WINDOW_NAME, parser);
    }

    return 0;
}