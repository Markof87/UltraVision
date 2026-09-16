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
    if (input.channels() == 3){
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else if (input.channels() == 4){
        cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
    }
    else{
        gray = input;
    }

    /*
    background uniformization
    */
    cv::Mat blurred_frame;
    cv::GaussianBlur(gray, blurred_frame, cv::Size(95, 95), 0, 0);

    cv::Mat grayFloat, blurredFloat, correctedFloat;
    gray.convertTo(grayFloat, CV_32FC1);                                //convert pixel values in floating number to perform a division operation
    blurred_frame.convertTo(blurredFloat, CV_32FC1);

    double mean_background = cv::mean(gray)[0];                         //compute the mean pixel value of the background

    cv::divide(grayFloat, blurredFloat, correctedFloat);                //applying a uniformization of darker areas in the background, toward lighter uniform background
    correctedFloat = correctedFloat * mean_background;                  //uniformization formula = "grayscale frame" / "blurred frame" * "mean background"

    cv::Mat final_gray;
    correctedFloat.convertTo(final_gray, CV_8UC1);                      //re-convert pixel values to 8-bit representation
    
    /*
    after the previous pre-elaboration, the proper segmentation phase starts
    */
    cv::Mat blurred;

    // takes as input the original frame and blurs it
    cv::GaussianBlur(final_gray, blurred, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

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
    cv::Mat v_closing_se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 17));       //if needed, change this structuring element size to have a "shorter" silhouette
    
    // performs a "vertical closing" to obtain a more "plain" silhouette
    cv::morphologyEx(otsu_sub, almost_segmented, cv::MORPH_CLOSE, v_closing_se);

    cv::Mat segmented;

    // (smoothing) elliptical structuring element
    cv::Mat smoothing_se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    // performs an additional smoothing to obtain a more defined silhouette
    cv::morphologyEx(almost_segmented, segmented, cv::MORPH_CLOSE, smoothing_se);

    output = segmented;

    /*these instructions are useful only for testing, to remove before submitting the solution
    cv::imshow("Segmentation", output);

    int key = cv::waitKey(1);
   */
}
