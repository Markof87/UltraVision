//WARNING: THIS IS JUST A TEST FILE, DO NOT USE IT IN THE FINAL VERSION

#include "TestingFunctions.h"
#include "SegmentationTechniques.h"

cv::Mat TestingFunctions::test_otsu_four_regions(const cv::Mat& frame) 
{
    int rows = frame.rows;
    int cols = frame.cols;

    // 1. APPLICAZIONE DI OTSU SULLA GRIGLIA 2x2
    cv::Mat grid_otsu_res = cv::Mat::zeros(rows, cols, CV_8U);
    int mid_h = rows / 2;
    int mid_w = cols / 2;

    cv::Rect quadrants[4] = {
        cv::Rect(0, 0, mid_w, mid_h),                
        cv::Rect(mid_w, 0, cols - mid_w, mid_h),       
        cv::Rect(0, mid_h, mid_w, rows - mid_h),       
        cv::Rect(mid_w, mid_h, cols - mid_w, rows - mid_h) 
    };

    for (int q = 0; q < 4; ++q) {
        cv::Mat sub_img = frame(quadrants[q]);
        cv::Mat sub_thresh = SegmentationTechniques::otsu(sub_img);
        sub_thresh.copyTo(grid_otsu_res(quadrants[q]));
    }

    return grid_otsu_res;
}