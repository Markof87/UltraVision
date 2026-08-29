/**
 * @file SilhouetteExtractorInterface.h
 * @brief Header file for the SilhouetteExtractorInterface class, which is an interface for application-specific silhouette extraction.
 * @author Marco Fabiani
 * 
 * Defines the polymorph structure of segmentation modules.
 */

 #ifndef SILHOUETTE_EXTRACTOR_INTERFACE_H
 #define SILHOUETTE_EXTRACTOR_INTERFACE_H

 #include <opencv2/opencv.hpp>

 /**
  * @class SilhouetteExtractorInterface
  * @brief Abstract base class for silhouette extraction techniques (interface).
  */

class SilhouetteExtractorInterface
{
  public:
      /**
       * @brief Virtual destructor for the SilhouetteExtractorInterface class.
       */
      virtual ~SilhouetteExtractorInterface() {};

      /**
       * @brief Pure virtual function to extract a silhouette from a given frame.
       * @param frame The input frame from which to extract the silhouette.
       * @return A binary image representing the extracted silhouette.
       */
      virtual cv::Mat extractSilhouette(const cv::Mat& frame) = 0;
};

#endif // SILHOUETTE_EXTRACTOR_INTERFACE_H