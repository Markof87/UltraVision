/** 
* @brief    Header for segmentation algorithm
* @details  The segmentation algorithm is in "segmentation_alg.cpp"
* @author   Matteo Longo 
*/


#ifndef SEGMENTATION_ALG
#define SEGMENTATION_ALG

#include <opencv2/opencv.hpp>

class SegmentationAlg{

    public:
        /**
        * @brief Segmentation function
        * @details This function performs segmentation on the input image and produces an output image.
        * @param input The input image to be segmented.
        * @param output The output image after segmentation.
        */
        void Segmentation(const cv::Mat& input, cv::Mat& output);

    private:
        cv::Mat prev_gray;   

};

#endif // SEGMENTATION_ALG
