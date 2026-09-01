/**
 * @file Utils.cpp
 * @brief Implementation of the Utils namespace, which contains utility functions for image processing.
 * @author Marco Fabiani
 */

#include <filesystem>

#include "utils/Utils.h"
#include "Sequencer.h"

namespace Utils 
{
    Sequencer sequencer = Sequencer();

    void loadSequenceFromCommandLineArguments(std::vector<cv::Mat>& loadedSequence, const std::string& BASE_PATH, const std::string& WINDOW_NAME, cv::CommandLineParser& parser)
    {
        std::string action;
        int personId;
        int scenarioId;

        // Load command line argument parameters
        loadArgumentParameters(parser, action, personId, scenarioId);

        // Load the dataset based on the provided command line parameters
        loadedSequence = loadDataset(BASE_PATH, action, personId, scenarioId);

        if(loadedSequence.empty())
        {
            std::cerr << "No images found for the specified parameters." << std::endl;
            return;
        }

        showDataset(loadedSequence, WINDOW_NAME);
    }

    void loadArgumentParameters(cv::CommandLineParser& parser, std::string& action, int& personId, int& scenarioId)
    {
        const std::string actions[] = {"boxing", "handclapping", "handwaving", "jogging", "running", "walking"};

        action = parser.get<std::string>("action");
        personId = parser.get<int>("person");
        scenarioId = parser.get<int>("scenario");

        // Check if the command line arguments are valid
        if(!parser.check())
        {
            parser.printErrors();
        }

        // if the lowercase of action is not in the list of actions, print an error message and exit
        if(!action.empty() && std::find(std::begin(actions), std::end(actions), action) == std::end(actions))
        {
            std::cerr << "Invalid action name: " << action << std::endl;
        }

        // if personId is not 1, 2, or 3, print an error message and exit
        if(personId != 0 && (personId < 1 || personId > 3))
        {
            std::cerr << "Invalid person ID: " << personId << std::endl;
        }

        // if scenarioId is not 1, 2, 3, or 4, print an error message and exit
        if(scenarioId != 0 && (scenarioId < 1 || scenarioId > 4))
        {
            std::cerr << "Invalid scenario ID: " << scenarioId << std::endl;
        }
    }

    std::vector<cv::Mat> loadDataset(const std::string& basePath, const std::string& actionName, int personId, int scenarioId)
    {
        std::vector<cv::Mat> dataset;

        // if no action is specified, load all actions for all persons and scenarios
        if(actionName.empty())
        {
            std::string actions[] = {"boxing", "handclapping", "handwaving", "jogging", "running", "walking"};
            for (const auto &action : actions) 
                for (int i = 1; i <= 3; i++) 
                    for (int j = 1; j <= 4; j++) 
                        addResource(dataset, basePath, action, i, j);
        }

        // if an action is specified but no person or scenario, load all persons and scenarios for that action
        if(!actionName.empty() && personId == 0 && scenarioId == 0)
        {
            for (int i = 1; i <= 3; i++) 
                for (int j = 1; j <= 4; j++) 
                    addResource(dataset, basePath, actionName, i, j);
        }

        //if an action and person are specified but no scenario, load all scenarios for that action and person
        if(!actionName.empty() && personId != 0 && scenarioId == 0)
        {
            for (int j = 1; j <= 4; j++) 
                addResource(dataset, basePath, actionName, personId, j);
        }

        //if an action, person and scenario are specified, load the dataset for that action, person and scenario
        if(!actionName.empty() && personId != 0 && scenarioId != 0)
        {
            addResource(dataset, basePath, actionName, personId, scenarioId);
        }

        return dataset;
    }

    void showDataset(const std::vector<cv::Mat>& dataset, const std::string& windowName) 
    {
        for (size_t i = 0; i < dataset.size(); ++i) 
        {
            cv::Mat imgResize;
            cv::resize(dataset[i], imgResize, cv::Size(600, 600), 0, 0, cv::INTER_NEAREST); 

            cv::imshow(windowName, imgResize);
            
            // Wait for 40ms (25 FPS) before showing the next image
            cv::waitKey(40);
        }
        
        char k = cv::waitKey(0);
    }

    void addResource(std::vector<cv::Mat>& dataset, const std::string& basePath, const std::string& action, int personId, int scenarioId) 
    {
        std::string path = basePath + "/" + action + "/person0" + std::to_string(personId) + "_" + action + "_d" + std::to_string(scenarioId) + "/data";
        for (const auto& entry : std::filesystem::directory_iterator(path)) 
        {
            cv::Mat image = cv::imread(entry.path().string());
            if (!image.empty()) 
                dataset.push_back(image);
        }
    }
}