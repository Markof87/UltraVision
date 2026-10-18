/**
 * @file performance_metrics.cpp
 * @brief Implementation of localization performance metrics.
 * @author Ivan Arduini
 */

#include "performance_metrics.h"
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

void PerformanceMetrics::reset() {
    records.clear();
}