//WARNING: THIS IS JUST A TEST FILE, DO NOT USE IT IN THE FINAL VERSION

#ifndef TESTING_FUNCTIONS_H
#define TESTING_FUNCTIONS_H

#include "FilterTechniques.h"
#include "SegmentationTechniques.h"

class TestingFunctions 
{
    public:
        static cv::Mat test_otsu_four_regions(const cv::Mat& frame);
};
#endif // TESTING_FUNCTIONS_H