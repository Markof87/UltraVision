/** 
* @brief    Header for bounding box drawing
* @details  The algorithm performing bounding box drawing is in "Bbox.cpp"
* @author   Matteo Longo 
*/


#ifndef B_BOX
#define B_BOX

#include <opencv2/opencv.hpp>

class Bbox
{
public:
    /**
     * @brief Bounding box drawing function
     * @details This function draws a bounding box around the actor silhouette on the original frame.
     * @param input The input segmented image.
     * @param output The output image (the original frame) containing the bounding box.
     */
    static void bbox_func(const cv::Mat& input, cv::Mat& output);
};

#endif // B_BOX