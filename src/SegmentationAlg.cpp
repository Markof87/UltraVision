/** 
* @brief    Segmentation algorithm
* @details  Based on Thresholded Bottom-Hat Transformation technique, it performs the segmentation of the frame.
* @author   Matteo Longo 
*/


#include <SegmentationAlg.h>


void segmentation(const cv::Mat& input, const cv::Mat& output){                         /// segmentation function

cv::Mat blurred;
cv::GaussianBlur(input, blurred, cv::Size(3, 3));                                       /// takes as input the original frame and blurs it

cv::Mat closed;
cv::Mat closing_se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(27, 25))     /// (closing) elliptical structuring element                     
cv::morphologyEx(blurred, closed, cv::MORPH_CLOSE, closing_se);                         /// takes as input the blurred frame and "closes" it with the structuring element "closing_se"

cv::Mat subtracted;
cv::subtract(closed, blurred, subtracted);                                              /// implements Bottom-Hat Transformation: subtracts the the frame (blurred) from its closing (closed).  
                                                                                        /// subtracted = closed - blurred.
cv::Mat otsu_sub;
cv::threshold(subtracted, otsu_sub, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);   /// produces the optimal thresolded (Otsu's) "subtracted" frame

cv::Mat almost_segmented;
cv::Mat v_closing_se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 11))       /// rectangular structuring element
cv::morphologyEx(otsu_sub, almost_segmented, cv::MORPH_CLOSE, v_closing_se);            /// performs a "vertical closing" to obtain a more "plain" silhouette

cv::Mat segmented;
cv:: Mat smoothing_se = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3))    /// (smoothing) elliptical structuring element
cv::morphologyEx(almost_segmented, segmented, cv::MORPH_CLOSE, smoothing_se);           /// performs an additional smoothing to obtain a more defined silhouette

}
