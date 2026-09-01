/**
 * @file SilhouetteExtractor.h
 * @brief Header file for the SilhouetteExtractor class, which provides functionality to extract silhouettes from frames using several techniques.
 * @author Marco Fabiani
 * 
 */

#ifndef SILHOUETTEEXTRACTOR_H
#define SILHOUETTEEXTRACTOR_H

#include "SilhouetteExtractorInterface.h"

/**
 * @class SilhouetteExtractor
 * @brief Concrete implementation of the SilhouetteExtractorInterface class.
 * 
 * This class provides functionality to extract silhouettes from frames using various techniques.
 */
class SilhouetteExtractor : public SilhouetteExtractorInterface
{
    public:
        /**
         * @brief Constructor for the SilhouetteExtractor class.
         */
        SilhouetteExtractor() {};

        /**
         * @brief Destructor for the SilhouetteExtractor class.
         */
        virtual ~SilhouetteExtractor() {};

        /**
         * @brief Implementation of the extractSilhouette function.
         * @param frame The input frame from which to extract the silhouette.
         * @return A binary image representing the extracted silhouette.
         */
        virtual cv::Mat extractSilhouette(const cv::Mat& frame) override;

    private:
        /**
         * @brief Helper function to perform background subtraction on the input frame.
         * @param frame The input frame from which to extract the silhouette.
         * @return A binary image representing the extracted silhouette after background subtraction.
         */
        cv::Mat backgroundSubtraction(const cv::Mat& frame);

        /**
         * @brief Helper function to perform thresholding on the input frame.
         * @param frame The input frame from which to extract the silhouette.
         * @return A binary image representing the extracted silhouette after thresholding.
         */
        cv::Mat thresholding(const cv::Mat& frame);
};

#endif // SILHOUETTEEXTRACTOR_H