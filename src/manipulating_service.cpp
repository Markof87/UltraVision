/**
 * @file manipulating_service.cpp
 * @brief Implementation of the ManipulatingService class.
 * @author Marco Fabiani
 */

#include "manipulating_service.h"
#include "segmentation_alg.h"
#include "bbox.h"
#include "classification.h"  //ADDED
#include "sequence_utils.h"  //ADDED


void ManipulatingService::manipulateFrame(const cv::Mat& inputFrame, cv::Mat& outputMask)
{
    if (inputFrame.empty()) return;

    static SegmentationAlg seg_alg; 
    seg_alg.Segmentation(inputFrame, outputMask);

    cv::Mat bboxFrame = inputFrame.clone();
    Bbox::bbox_func(outputMask, bboxFrame);

    outputMask = bboxFrame;
    // outputMask = inputFrame.clone(); // For now, just clone the input frame to the output mask
}


//ADDED
std::string ManipulatingService::manipulateSequence(const std::vector<cv::Mat>& sequenceFrames, 
                                                   std::vector<cv::Mat>& annotatedFrames)
{
    if (sequenceFrames.empty()) return "";

    std::vector<cv::Mat> mask_sequence;
    std::vector<cv::Rect> bbox_sequence;
    std::vector<cv::Mat> frames_with_bbox;

    mask_sequence.reserve(sequenceFrames.size());
    bbox_sequence.reserve(sequenceFrames.size());
    frames_with_bbox.reserve(sequenceFrames.size());

    static SegmentationAlg seg_alg;

    // 1. Process each frame: compute mask, bounding box, and visual rendering
    for (const auto& raw_frame : sequenceFrames) {
        if (raw_frame.empty()) continue;

        cv::Mat mask;
        seg_alg.Segmentation(raw_frame, mask);

        cv::Mat frame_with_bbox = raw_frame.clone();
        cv::Rect bbox = Bbox::bbox_func(mask, frame_with_bbox);

        mask_sequence.push_back(mask);
        bbox_sequence.push_back(bbox);
        frames_with_bbox.push_back(frame_with_bbox);
    }

    // 2. Extract temporal features across the whole sequence and classify the action
    SequenceFeatures dyn_feats = extractSequenceFeatures(mask_sequence, bbox_sequence);
    std::string predicted_action = classifySequence(dyn_feats, mask_sequence, bbox_sequence);

    // 3. Clone frames and overlay the predicted action label as red text in the bottom-left corner
    annotatedFrames = overlayActionOnFrames(frames_with_bbox, predicted_action);

    return predicted_action;
}