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


bool PerformanceMetrics::loadGroundTruth(const std::string& filepath, GroundTruthAnnotation& gt) {
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

    gt.class_id = class_id;

    // Convert center format (x_c, y_c, w, h) into top-left format (x, y, w, h) for cv::Rect2f
    float top_left_x = x_center - (width / 2.0f);
    float top_left_y = y_center - (height / 2.0f);

    gt.bounding_box = cv::Rect2f(top_left_x, top_left_y, width, height);

    return true;
}

void PerformanceMetrics::addRecord(const SequenceEvaluationRecord& record) {
    records.push_back(record);
}

//=====================================================================================================================================
//      LOCALIZATION METRICS (BOUNDING BOX)
//=====================================================================================================================================
double PerformanceMetrics::computeIoU(const cv::Rect2f& bboxA, const cv::Rect2f& bboxB) {
    cv::Rect2f intersection = bboxA & bboxB;  // Determines overlapping intersection rectangle
    float intersection_area = intersection.area();

    float union_area = bboxA.area() + bboxB.area() - intersection_area;  // Calculate union area


    if (union_area <= 0.0f) {
        return 0.0;
    }

    return static_cast<double>(intersection_area / union_area);  // Returns the IoU score
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


//=====================================================================================================================================
//      CLASSIFICATION METRICS (CATEGORY)
//=====================================================================================================================================
double PerformanceMetrics::computeGlobalAccuracy() const {
    if (records.empty()) {
        return 0.0;
    }

    int count = 0;
    
    for (const auto& record : records) {
        if (record.is_correct) {
            count++;
        }
    }

    // Return the percentage of correct predictions over the total sequences
    return (static_cast<double>(count) / records.size()) * 100.0;
}

std::vector<double> PerformanceMetrics::computeF1Scores() const {
    std::vector<double> f1_scores(6, 0.0);
    
    for (int i = 0; i < 6; ++i) {
        int class_id = i + 1;
        int true_positives = 0;
        int false_positives = 0;
        int false_negatives = 0;

        for (const auto& record : records) {
            // Check if the actual ground truth action matches the current loop class
            bool belongs_to_class = (record.ground_truth_class == class_id);
            // Check if our predicted action matches the current loop class
            bool is_predicted_as_class = (record.predicted_class == class_id);

            if (belongs_to_class && is_predicted_as_class) {
                true_positives++;
            } else if (!belongs_to_class && is_predicted_as_class) {
                false_positives++;
            } else if (belongs_to_class && !is_predicted_as_class) {
                false_negatives++;
            }
        }

        double precision = 0.0;  // precision = TP / (TP + FP)
        double recall = 0.0;     // recall = TP / (TP + FN)

        if ((true_positives + false_positives) > 0) {
            precision = static_cast<double>(true_positives) / (true_positives + false_positives);
        }

        if ((true_positives + false_negatives) > 0) {
            recall = static_cast<double>(true_positives) / (true_positives + false_negatives);
        }

        if ((precision + recall) > 0.0) {
            f1_scores[i] = 2.0 * (precision * recall) / (precision + recall);
        }
    }

    return f1_scores;
}

std::vector<std::vector<int>> PerformanceMetrics::computeConfusionMatrix() const {
    // Initialize a 6x6 matrix with zeros (rows: Ground Truth, cols: Predicted)
    std::vector<std::vector<int>> confusion_matrix(6, std::vector<int>(6, 0));

    // Populate the confusion matrix based on the recorded evaluations
    for (const auto& record : records) {
        // Ensure class IDs are within the valid range [1, 6]
        if (record.ground_truth_class >= 1 && record.ground_truth_class <= 6 &&
            record.predicted_class >= 1 && record.predicted_class <= 6) {
            
            int gt_idx = record.ground_truth_class - 1;
            int pred_idx = record.predicted_class - 1;
            
            confusion_matrix[gt_idx][pred_idx]++;
        }
    }

    return confusion_matrix;
}


//=====================================================================================================================================
//      EXPORT FUNCTIONS
//=====================================================================================================================================
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
    double global_accuracy = computeGlobalAccuracy();
    std::vector<double> f1_scores = computeF1Scores();
    std::vector<std::vector<int>> confusion_matrix = computeConfusionMatrix();

    csv_file << "\n# --- SUMMARY METRICS ---\n";

    csv_file << "meanIoU,,,,,," << std::fixed << std::setprecision(4) << miou << "\n";

    csv_file << "global_accuracy_percent,,,,,," << std::fixed << std::setprecision(2) << global_accuracy << "\n";

    csv_file << "\n# --- PER-CLASS F1-SCORES ---\n";
    csv_file << "class_id,class_name,f1_score\n";
    std::vector<std::string> class_names = {"boxing", "handclapping", "handwaving", "jogging", "running", "walking"};
    for (size_t i = 0; i < f1_scores.size(); ++i) {
        csv_file << (i + 1) << "," << class_names[i] << "," << std::fixed << std::setprecision(4) << f1_scores[i] << "\n";
    }

    csv_file << "\n# --- CONFUSION MATRIX (Rows: GT, Cols: Pred) ---\n";
    csv_file << ",Pred_C1,Pred_C2,Pred_C3,Pred_C4,Pred_C5,Pred_C6\n";
    for (int i = 0; i < 6; ++i) {
        csv_file << "GT_C" << (i + 1);
        for (int j = 0; j < 6; ++j) {
            csv_file << "," << confusion_matrix[i][j];
        }
        csv_file << "\n";
    }
    return true;
}

bool PerformanceMetrics::exportToTXT(const std::string& filepath) const {
    std::ofstream txt_file(filepath);
    if (!txt_file.is_open()) {
        std::cerr << "[PerformanceMetrics] Error: Unable to create TXT report file at " << filepath << std::endl;
        return false;
    }

    double miou = calculateMeanIoU();
    double global_accuracy = computeGlobalAccuracy();
    std::vector<double> f1_scores = computeF1Scores();
    std::vector<std::vector<int>> confusion_matrix = computeConfusionMatrix();

    txt_file << "=====================================================\n";
    txt_file << "                PERFORMANCE REPORT                   \n";
    txt_file << "=====================================================\n\n";

    txt_file << "Total Evaluated Sequences : " << records.size() << "\n";
    txt_file << "Mean IoU (Frame 20)       : " << std::fixed << std::setprecision(4) << miou << "\n";
    txt_file << "Global Accuracy           : " << std::fixed << std::setprecision(2) << global_accuracy << "%\n\n";
    
    txt_file << "-----------------------------------------------------\n";
    txt_file << " Per-Class F1-Scores:                                \n";
    txt_file << "-----------------------------------------------------\n";
    std::vector<std::string> class_names = {"Boxing", "Handclapping", "Handwaving", "Jogging", "Running", "Walking"};
    for (size_t i = 0; i < f1_scores.size(); ++i) {
        txt_file << "  - " << std::left << std::setw(15) << class_names[i] 
                 << " (Class " << (i + 1) << ") : " << std::fixed << std::setprecision(4) << f1_scores[i] << "\n";
    }
    txt_file << "\n";

    txt_file << "-----------------------------------------------------\n";
    txt_file << " Confusion Matrix (Rows: GT, Cols: Predictions 1-6)  \n";
    txt_file << "-----------------------------------------------------\n";
    
    // Print column headers with fixed width
    txt_file << "      ";
    for (int j = 1; j <= 6; ++j) {
        txt_file << std::setw(6) << ("C" + std::to_string(j));
    }
    txt_file << "\n";

    // Print rows with fixed width for each cell
    for (int i = 0; i < 6; ++i) {
        txt_file << "C" << (i + 1) << "  | ";
        for (int j = 0; j < 6; ++j) {
            txt_file << std::setw(6) << confusion_matrix[i][j];
        }
        txt_file << "\n";
    }
    txt_file << "\n";

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

    const int total_sequences = 72;
    std::cout << "[PerformanceMetrics] Evaluation started. Processing " << total_sequences << " valid sequences...\n";

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

            processed_sequences_count++;
            std::cout << "[Evaluation] Processing valid sequence [" << processed_sequences_count << "/" << total_sequences 
                      << "] -> Action: " << std::left << std::setw(15) << action_name 
                      << " | ID: " << std::left << std::setw(25) << seq_id << "\r" << std::flush;

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
            if (action_name == "boxing") gt_class_id = 1;
            else if (action_name == "handclapping") gt_class_id = 2;
            else if (action_name == "handwaving") gt_class_id = 3;
            else if (action_name == "jogging") gt_class_id = 4;
            else if (action_name == "running") gt_class_id = 5;
            else if (action_name == "walking") gt_class_id = 6;
                
            if (loadGroundTruth(gt_filepath, gt)) {
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

            int pred_class_id = 0;
            if (predicted_action == "boxing" || predicted_action == "Boxing") pred_class_id = 1;
            else if (predicted_action == "handclapping" || predicted_action == "Handclapping") pred_class_id = 2;
            else if (predicted_action == "handwaving" || predicted_action == "Handwaving") pred_class_id = 3;
            else if (predicted_action == "jogging" || predicted_action == "Jogging") pred_class_id = 4;
            else if (predicted_action == "running" || predicted_action == "Running") pred_class_id = 5;
            else if (predicted_action == "walking" || predicted_action == "Walking") pred_class_id = 6;

            record.predicted_class = pred_class_id;

            record.is_correct = (record.action_name == action_name);
            record.iou_median_frame = current_iou; 

            addRecord(record);
        }
    }

    std::cout << "\n[PerformanceMetrics] Batch evaluation completed successfully. Generating reports..." << std::endl;

    // Export results to both CSV and TXT files
    bool csv_success = exportToCSV(csvFilePath);
    bool txt_success = exportToTXT(txtFilePath);

    if (csv_success && txt_success) {
        std::cout << "[PerformanceMetrics] Reports exported successfully to TXT and CSV." << std::endl;
    }

    return csv_success && txt_success;
}

void PerformanceMetrics::reset() {
    records.clear();
}