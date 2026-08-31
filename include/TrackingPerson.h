/**
 * @file TrackingPerson.h
 * @brief Header file for the TrackingPerson class, which provides functionality to track the person in a frame.
 * @author Marco Fabiani
 * 
 */

#ifndef TRACKINGPERSON_H
#define TRACKINGPERSON_H

#include <opencv2/opencv.hpp>

/**
 * @class TrackingPerson
 * @brief A class that provides functionality to track the person in a frame.
 */

class TrackingPerson
{
    public:
        /**
         * @brief Constructor for the TrackingPerson class.
         */
        TrackingPerson() {};

        /**
         * @brief Destructor for the TrackingPerson class.
         */
        virtual ~TrackingPerson() {};

        /**
         * @brief Tracks the person in the given frame.
         * @param frame The input frame in which to track the person.
         * @return A cv::Rect representing the bounding box of the tracked person.
         */
        cv::Rect getActorBoundingBox(const cv::Mat& frame);
};

#endif // TRACKINGPERSON_H