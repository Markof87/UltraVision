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
    a conditional elaboration that provides a frame brighter at its borders (according to 'border' areas) and equal to the original frame in the central area,
    in order to reduce shadows noise at the borders of the frame, without modifying the actor in sequences where the actor "appears" from a side of the frame and disappears in the opposite side
    */
    cv::Mat final_gray = gray.clone();

    if(!prev_gray.empty()){

        int border = 30;  //create two rectangle areas situated at both lateral borders in the frame, which width is 30 pixels
        int motionThresh = 15;  //a measurement of likelihood of 'motion' in those lateral areas, given by the difference between two subsequent frames
        int pixelThresh = 50;   //number of pixels variation, in the actual frame ('gray'), from the previous frame ('prev_gray')

        cv::Mat prev_diff;
        cv::absdiff(gray, prev_gray, prev_diff);    //performs thesubtraction between 2 subsequent frames
        cv::threshold(prev_diff, prev_diff, motionThresh, 255, cv::THRESH_BINARY);  //pixels with intensity > 'motionThresh' (pixels involved in motion) became white

        cv::Rect leftBorderRect(0, 0, border, gray.rows);
        cv::Rect rightBorderRect(gray.cols - border, 0, border, gray.rows);

        cv::Mat leftBorder = prev_diff(leftBorderRect);
        cv::Mat rightBorder = prev_diff(rightBorderRect);

        int leftMotion = cv::countNonZero(leftBorder);
        int rightMotion = cv::countNonZero(rightBorder);

        if(leftMotion <= pixelThresh && rightMotion >= pixelThresh){    //if no sufficient motion is detected in both borders areas

            cv::Mat variationMap = cv::Mat::zeros(gray.rows, gray.cols, CV_32FC1);
            
            /*
            it needs a "new" reference system with the center (0;0) of (normalized) coordinates (x,y) lying in the geometrical center of the image
            in order to compute a blurring filter acting on borders of the image but not in the central area of the image (frame)
            */
            for(int y = 0; y < gray.rows; ++y){
                for(int x = 0; x < gray.cols; ++x){

                    float x0 = 2.0f * x / (gray.cols - 1) - 1.0f;           //normalized coordinates (-1 at leftmost and top; 0 at the center; 1 at rightmost and bottom)
                    float y0 = 2.0f * y / (gray.rows - 1) - 1.0f;
                    float distance = std::max(std::abs(x0), std::abs(y0));  //take maximum value among vertical and horizontal distance from borders

                    variationMap.at<float>(y, x) = std::pow(distance, 4);   //gradually fade pixels in borders areas following a "power-of-4 curve trend"
                }
            }
                
            cv::Mat imgFloat;
            gray.convertTo(imgFloat, CV_32FC1);                                 //convert the pixels of the frame in floating point values, in order to avoid computation problems

            float light_param = 80.0f;                                          //define a "lightning" parameter

            cv::Mat finalFloat = imgFloat + (variationMap * light_param);       //apply "lightning" transformation: more external pixels became lighter, central pixel remain the same
            finalFloat.convertTo(final_gray, CV_8UC1);                          //come back to the "standard" 8-bit pixel values
        }
    }

    prev_gray = gray.clone();   //current frame is the previous frame for the next iteration

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
