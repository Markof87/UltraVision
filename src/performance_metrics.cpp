/**
 * @file performance_metrics.cpp
 * @brief Implementation of localization performance metrics.
 * @author Ivan Arduini
 */

#include "performance_metrics.h"
#include "manipulating_service.h"
#include "segmentation_alg.h"
#include "bbox.h"
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

double PerformanceMetrics::computeIoU(const cv::Rect2f& bboxA, const cv::Rect2f& bboxB) {
    cv::Rect2f intersection = bboxA & bboxB;  // Determines overlapping intersection rectangle
    float intersection_area = intersection.area();

    float union_area = bboxA.area() + bboxB.area() - intersection_area;  // Calculate union area


    if (union_area <= 0.0f) {
        return 0.0;
    }

    return static_cast<double>(intersection_area / union_area);  // Returns the IoU score
}

bool PerformanceMetrics::loadGroundTruth(const std::string& filepath, GroundTruthAnnotation& annotation) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[PerformanceMetrics] Error: Unable to open ground truth file at " << filepath << std::endl;
        return false;
    }

    int class_id = 0;
    float x_center = 0.0f;
    float y_center = 0.0f;
    float width = 0.0f;
    float height = 0.0f;

    // Read format: <class_id> <x_center> <y_center> <width> <height>
    if (!(file >> class_id >> x_center >> y_center >> width >> height)) {
        std::cerr << "[PerformanceMetrics] Error: Wrong ground truth file format in " << filepath << std::endl;
        return false;
    }

    annotation.class_id = class_id;

    // Convert center format (x_c, y_c, w, h) into top-left format (x, y, w, h) for cv::Rect2f
    float top_left_x = x_center - (width / 2.0f);
    float top_left_y = y_center - (height / 2.0f);

    annotation.bounding_box = cv::Rect2f(top_left_x, top_left_y, width, height);

    return true;
}

void PerformanceMetrics::addRecord(const SequenceEvaluationRecord& record) {
    records.push_back(record);
}

double PerformanceMetrics::calculateMeanIoU() const {
    if (records.empty()) {
        return 0.0;
    }

    double total_iou = 0.0;
    for (const auto& record : records) {
        total_iou += record.iou_median_frame;
    }

    return total_iou / static_cast<double>(records.size());
}

bool PerformanceMetrics::exportToCSV(const std::string& filepath) const {
    std::ofstream csv_file(filepath);
    if (!csv_file.is_open()) {
        std::cerr << "[PerformanceMetrics] Error: Unable to create CSV file at " << filepath << std::endl;
        return false;
    }

    csv_file << "sequence_id,action_name,ground_truth_class,predicted_class,is_correct,iou_median_frame\n";

    for (const auto& record : records) {
        csv_file << record.sequence_id << ","
                 << record.action_name << ","
                 << record.ground_truth_class << ","
                 << record.predicted_class << ","
                 << (record.is_correct ? 1 : 0) << ","
                 << std::fixed << std::setprecision(4) << record.iou_median_frame << "\n";
    }

    double miou = calculateMeanIoU();
    csv_file << "meanIoU,,,,,," << std::fixed << std::setprecision(4) << miou << "\n";

    return true;
}

bool PerformanceMetrics::exportToTXT(const std::string& filepath) const {
    std::ofstream txt_file(filepath);
    if (!txt_file.is_open()) {
        std::cerr << "[PerformanceMetrics] Error: Unable to create TXT report file at " << filepath << std::endl;
        return false;
    }

    double miou = calculateMeanIoU();

    txt_file << "=====================================================\n";
    txt_file << "               LOCALIZATION REPORT                   \n";
    txt_file << "=====================================================\n\n";
    txt_file << "Total Evaluated Sequences : " << records.size() << "\n";
    txt_file << "Mean IoU (Frame 20)       : " << std::fixed << std::setprecision(4) << miou << "\n\n";
    txt_file << "-----------------------------------------------------\n";
    txt_file << " Per-Sequence IoU Breakdown:                         \n";
    txt_file << "-----------------------------------------------------\n";

    for (const auto& record : records) {
        txt_file << "[" << record.sequence_id << "] "
                 << "Action: " << std::left << std::setw(14) << record.action_name
                 << " | IoU (Frame 20): " << std::fixed << std::setprecision(4) << record.iou_median_frame
                 << "\n";
    }

    return true;
}

bool PerformanceMetrics::exportCompleteEvaluation(const std::string& csvFilePath, const std::string& txtFilePath) {
    reset(); // Clear any previous records to ensure a new global evaluation

    std::string base_sequences_path = "../data/Sequences";
    if (!std::filesystem::exists(base_sequences_path)) {
        std::cerr << "[PerformanceMetrics] Error: Sequences base path not found at " << base_sequences_path << std::endl;
        return false;
    }

    ManipulatingService manipulator;
    int processed_sequences_count = 0;

    // Iteration through each action directory (e.g. boxing, walking, etc.)
    for (const auto& action_entry : std::filesystem::directory_iterator(base_sequences_path)) {
        if (!action_entry.is_directory()) continue; // Skip non-directory entries
        std::string action_name = action_entry.path().filename().string();

        // Iteration through each sequence instance for the current action (e.g. p1_d1, ..., p2_d1, ..., p3_d4)
        for (const auto& seq_entry : std::filesystem::directory_iterator(action_entry.path())) {
            if (!seq_entry.is_directory()) continue; // Skip non-directory entries
            std::string seq_id = seq_entry.path().filename().string();
            std::string seq_data_path = seq_entry.path().string() + "/data";

            if (!std::filesystem::exists(seq_data_path)) continue;

            // Frame collection (sorting frame paths alphabetically)
            std::vector<std::string> frame_paths;
            for (const auto& frame_file : std::filesystem::directory_iterator(seq_data_path)) {
                frame_paths.push_back(frame_file.path().string());
            }
            std::sort(frame_paths.begin(), frame_paths.end());


            // Load sorted frames into memory
            std::vector<cv::Mat> sequence_frames;
            sequence_frames.reserve(frame_paths.size());
            for (const auto& path : frame_paths) {
                cv::Mat frame = cv::imread(path);
                if (!frame.empty()) {
                    sequence_frames.push_back(frame);
                }
            }

            if (sequence_frames.empty()) continue;

            // Sequence processing (using the manipulation service)
            std::vector<cv::Mat> annotated_frames;
            std::string predicted_action = manipulator.manipulateSequence(sequence_frames, annotated_frames);

            GroundTruthAnnotation gt;
            std::string gt_filepath = seq_entry.path().string() + "/labels/ground_truth.txt";            
            double current_iou = 0.0;
            int gt_class_id = 0;

            if (loadGroundTruth(gt_filepath, gt)) {
                gt_class_id = gt.class_id;

                cv::Mat frame_20 = sequence_frames[19];
                cv::Mat mask_20;
                static SegmentationAlg seg_alg;
                seg_alg.Segmentation(frame_20, mask_20);
                cv::Mat dummy_frame = frame_20.clone();
                cv::Rect raw_bbox = Bbox::bbox_func(mask_20, dummy_frame);

                // Convert cv::Rect to cv::Rect2f to match computeIoU signature
                cv::Rect2f predicted_box_at_20 = static_cast<cv::Rect2f>(raw_bbox);

                current_iou = computeIoU(predicted_box_at_20, gt.bounding_box);
            }

            // Build and add the evaluation record
            SequenceEvaluationRecord record;
            record.sequence_id = seq_id;
            record.action_name = predicted_action;

            record.ground_truth_class = gt_class_id; 
            record.predicted_class = 0;  
            record.is_correct = (record.action_name == action_name);
            record.iou_median_frame = current_iou; 

            addRecord(record);
            processed_sequences_count++;
        }
    }

    std::cout << "[PerformanceMetrics] Batch evaluation completed. Processed " 
              << processed_sequences_count << " sequences." << std::endl;

    // Export results to both CSV and TXT files
    bool csv_success = exportToCSV(csvFilePath);
    bool txt_success = exportToTXT(txtFilePath);

    return csv_success && txt_success;
}

void PerformanceMetrics::reset() {
    records.clear();
}