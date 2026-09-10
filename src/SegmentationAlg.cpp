/** 
* @brief    Segmentation algorithm
* @details  Based on Thresholded Bottom-Hat Transformation technique, it performs the segmentation of the frame.
* @author   Matteo Longo 
*/


#include "SegmentationAlg.h"

// Segmentation function
void SegmentationAlg::Segmentation(const cv::Mat& input, cv::Mat& output)
{                         
    cv::Mat gray;
    if (input.channels() == 3)
    {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else if (input.channels() == 4)
    {
        cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        gray = input;
    }

    cv::Mat blurred;

    /// takes as input the original frame and blurs it
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT); 

    cv::Mat closed;

    // (closing) elliptical structuring element 
    cv::Mat closing_se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(27, 25));     
    
    // takes as input the blurred frame and "closes" it with the structuring element "closing_se"
    cv::morphologyEx(blurred, closed, cv::MORPH_CLOSE, closing_se);

    cv::Mat subtracted;
    // implements Bottom-Hat Transformation: subtracts the the frame (blurred) from its closing (closed).  
    cv::subtract(closed, blurred, subtracted);
    
    // subtracted = closed - blurred.
    cv::Mat otsu_sub;

    // produces the optimal thresolded (Otsu's) "subtracted" frame
    cv::threshold(subtracted, otsu_sub, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat almost_segmented;
    
    // rectangular structuring element
    cv::Mat v_closing_se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 11));       
    
    // performs a "vertical closing" to obtain a more "plain" silhouette
    cv::morphologyEx(otsu_sub, almost_segmented, cv::MORPH_CLOSE, v_closing_se);

    cv::Mat segmented;

    // (smoothing) elliptical structuring element
    cv::Mat smoothing_se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    // performs an additional smoothing to obtain a more defined silhouette
    cv::morphologyEx(almost_segmented, segmented, cv::MORPH_CLOSE, smoothing_se);

    output = segmented;
}
