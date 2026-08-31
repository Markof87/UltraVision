/**
 * @file TrackingPerson.cpp
 * @brief Implementation file for the TrackingPerson class.
 * @author Marco Fabiani
 */

#include "TrackingPerson.h"

cv::Rect TrackingPerson::getActorBoundingBox(const cv::Mat& frame)
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // Find all external contours in the binary image
    cv::findContours(frame, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::Rect actorRect(0, 0, 0, 0);

    // If contours are found, find the largest contour and compute its bounding box
    if (!contours.empty())
    {
        double maxArea = 0.0;
        int maxAreaIndex = 0;

        for(size_t i = 0; i < contours.size(); ++i)
        {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea)
            {
                maxArea = area;
                maxAreaIndex = i;
            }
        }

        //Compute the minimum bounding rectangle for the actor's contour
        actorRect = cv::boundingRect(contours[maxAreaIndex]);
    }

    return actorRect;
}