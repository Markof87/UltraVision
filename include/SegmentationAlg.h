/** 
* @brief    Header for segmentation algorithm
* @details  The segmentation algorithm is in "segmentation_alg.cpp"
* @author   Matteo Longo 
*/


#ifndef SEGMENTATION_ALG
#define SEGMENTATION_ALG

#include <opencv2/opencv.hpp>


void segmentation(const cv::Mat& input, const cv::Mat& output);

#endif
