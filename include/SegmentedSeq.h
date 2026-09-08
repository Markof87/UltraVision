/**
* @brief    A data structure for storing the segmented sequence 
* @details  It represents an output vector from segmentation phase, to be passed to next phases
* @author   Matteo Longo
* @param    silhouette  stores the segmented frame (the silhouette)
* @param    action      stores the reference to the action involved in the original (and the segmented) frame
* @param    person      stores the reference to the person involved in the original (and the segmented) frame
* @param    scenario    stores the reference to the scenario involved in the original (and the segmented) frame
* @param    frame_n     stores the number of the original (and the segmented) frame
*/


#ifndef SEGMENTED_SEQ
#define SEGMENTED_SEQ

#include <opencv2/opencv.hpp>
#include <string>

struct SegmentedSeq {       
    cv::Mat silhouette;
    std::string action;
    std::string person;
    std::string scenario;
    std::string frame_n;
};

#endif