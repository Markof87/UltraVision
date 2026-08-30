/**
 * @file Utils.h
 * @brief Header file for the Utils namespace, which contains utility functions.
 * @author Marco Fabiani
 * 
 */

#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * @namespace Utils
 * @brief A namespace containing utility functions for image processing.
 */

namespace Utils
{
    /**
     * @brief Loads a dataset of images from a specified path.
     * @param basePath The base path to the dataset.
     * @param action The action name (e.g., "boxing", "handclapping").
     * @param personId The ID of the person (e.g., 1, 2, 3).
     * @param scenarioId The ID of the scenario (e.g., 1, 2, 3, 4).
     * @return A vector of cv::Mat objects representing the loaded images.
     */
    std::vector<cv::Mat> loadDataset(const std::string& basePath, const std::string& action, int personId, int scenarioId);

    /**
     * @brief Displays a dataset of images in a window.
     * @param dataset A vector of cv::Mat objects representing the images to be displayed.
     * @param windowName The name of the window in which to display the images.
     */
    void showDataset(const std::vector<cv::Mat>& dataset, const std::string& windowName);

    /**
     * @brief Adds a resource (image) to the dataset.
     * @param dataset The vector to which the image will be added.
     * @param basePath The base path to the dataset.
     * @param action The action name (e.g., "boxing", "handclapping").
     * @param personId The ID of the person (e.g., 1, 2, 3).
     * @param scenarioId The ID of the scenario (e.g., 1, 2, 3, 4).
     */
    void addResource(std::vector<cv::Mat>& dataset, const std::string& basePath, const std::string& action, int personId, int scenarioId);
}

#endif // UTILS_H