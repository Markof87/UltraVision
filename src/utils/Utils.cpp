/**
 * @file Utils.cpp
 * @brief Implementation of the Utils namespace, which contains utility functions for image processing.
 * @author Marco Fabiani
 */

#include <filesystem>

#include "utils/Utils.h"

namespace Utils 
{
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
        for (const auto& image : dataset) 
        {
            cv::Mat imgResize;
            cv::resize(image, imgResize, cv::Size(600, 600), 0, 0, cv::INTER_NEAREST); 

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