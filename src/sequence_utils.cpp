#include "sequence_utils.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <cmath>

namespace fs = std::filesystem;

std::vector<SequenceData> loadSequences(const std::string& main_folder) {
    std::vector<SequenceData> all_sequences;

    if (!fs::exists(main_folder) || !fs::is_directory(main_folder)) {
        std::cerr << "[ERROR] Main folder does not exist: " << main_folder << std::endl;
        return all_sequences;
    }

    std::cout << "[INFO] Scanning KTH dataset action sequences..." << std::endl;

    for (const auto& action_folder : fs::directory_iterator(main_folder)) {
        if (!action_folder.is_directory()) continue;
        for (const auto& seq_folder : fs::directory_iterator(action_folder.path())) {
            if (!seq_folder.is_directory()) continue;

            std::string data_path = seq_folder.path().string() + "/data/";
            if (!fs::exists(data_path)) continue;

            std::vector<std::string> image_files;
            for (const auto& entry : fs::directory_iterator(data_path)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    if (ext == ".png" || ext == ".jpg") {
                        image_files.push_back(entry.path().string());
                    }
                }
            }
            std::sort(image_files.begin(), image_files.end());

            if (!image_files.empty()) {
                SequenceData seq;
                seq.action_name = action_folder.path().filename().string();
                seq.seq_name = seq_folder.path().filename().string();
                seq.image_files = image_files;
                all_sequences.push_back(seq);
            }
        }
    }

    std::cout << "[INFO] Found " << all_sequences.size() << " total sequences." << std::endl;
    return all_sequences;
}

bool visualizeSequence(const std::vector<cv::Mat>& sequence_frames,
                       const std::vector<cv::Mat>& mask_sequence,
                       const std::vector<cv::Rect>& bbox_sequence,
                       const SequenceData& current_seq,
                       const std::string& predicted_action,
                       const SequenceFeatures& dyn_feats,
                       const StaticFeatures& static_feats,
                       const std::string& win_name,
                       bool& skip_visualization,
                       int& seq_index) {

    if (skip_visualization) return false;

    int frame_index = 0;
    bool paused = false;
    bool next_seq = false;
    bool prev_seq = false;

    // Check if the predicted action belongs to the dynamic or static branch
    bool is_dynamic_pred = (predicted_action == "running" || 
                            predicted_action == "jogging" || 
                            predicted_action == "walking");

    while (frame_index >= 0 && frame_index < static_cast<int>(sequence_frames.size())) {
        cv::Mat frame = sequence_frames[frame_index];
        cv::Rect bbox = bbox_sequence[frame_index];

        cv::Mat frame_color;
        cv::cvtColor(frame, frame_color, cv::COLOR_GRAY2BGR);
        if (bbox.width > 0 && bbox.height > 0) {
            cv::rectangle(frame_color, bbox, cv::Scalar(0, 0, 255), 2);
        }

        cv::Mat diff_color = cv::Mat::zeros(frame.size(), CV_8UC3);
        if (frame_index > 0) {
            cv::Mat diff;
            cv::absdiff(mask_sequence[frame_index - 1], mask_sequence[frame_index], diff);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::morphologyEx(diff, diff, cv::MORPH_OPEN, kernel);
            cv::cvtColor(diff, diff_color, cv::COLOR_GRAY2BGR);
            if (bbox.width > 0 && bbox.height > 0) {
                cv::rectangle(diff_color, bbox, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::Mat resized_frame, resized_diff;
        cv::resize(frame_color, resized_frame, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
        cv::resize(diff_color, resized_diff, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);

        // Explanatory side panel
        cv::Mat explanation_panel(resized_frame.rows, 320, CV_8UC3, cv::Scalar(40, 40, 40));
        
        // Header & Predictions
        cv::putText(explanation_panel, "ACTION DETAILS:", cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        cv::putText(explanation_panel, "Ground Truth: " + current_seq.action_name, cv::Point(15, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        
        cv::Scalar pred_color = (predicted_action == current_seq.action_name) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::putText(explanation_panel, "Predicted: " + predicted_action, cv::Point(15, 90), cv::FONT_HERSHEY_SIMPLEX, 0.55, pred_color, 2);

        // Conditional feature display based on action type
        if (is_dynamic_pred) {
            cv::putText(explanation_panel, "--- DYNAMIC FEATS ---", cv::Point(15, 130), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 0), 1);
            cv::putText(explanation_panel, "Disp: " + std::to_string(dyn_feats.net_displacement).substr(0, 5), cv::Point(15, 160), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1);
            cv::putText(explanation_panel, "AvgVel: " + std::to_string(dyn_feats.avg_velocity).substr(0, 4), cv::Point(15, 190), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1);
            cv::putText(explanation_panel, "Peaks: " + std::to_string(dyn_feats.stride_peaks), cv::Point(15, 220), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1);
        } else {
            cv::putText(explanation_panel, "--- STATIC FEATS ---", cv::Point(15, 130), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);
            cv::putText(explanation_panel, "Min CY: " + std::to_string(static_feats.min_cy).substr(0, 5), cv::Point(15, 160), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1);
            cv::putText(explanation_panel, "X Offset: " + std::to_string(static_feats.avg_x_offset).substr(0, 5), cv::Point(15, 190), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1);
            cv::putText(explanation_panel, "Y Range: " + std::to_string(static_feats.y_range).substr(0, 5), cv::Point(15, 220), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1);
        }

        std::string info_seq = "Seq: " + current_seq.action_name + "/" + current_seq.seq_name;
        std::string info_frame = "Frame: " + std::to_string(frame_index + 1) + "/" + std::to_string(sequence_frames.size()) + (paused ? " [PAUSED]" : "");
        cv::putText(resized_frame, info_seq, cv::Point(15, 25), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 2);
        cv::putText(resized_frame, info_frame, cv::Point(15, 45), cv::FONT_HERSHEY_SIMPLEX, 0.40, cv::Scalar(255, 255, 255), 1);

        cv::Mat temp_combined, triple_view;
        cv::hconcat(resized_frame, resized_diff, temp_combined);
        cv::hconcat(temp_combined, explanation_panel, triple_view);

        cv::imshow(win_name, triple_view);

        char key = static_cast<char>(cv::waitKey(paused ? 0 : 50));

        if (key == 'q' || key == 'Q') {
            skip_visualization = true;
            break;
        } else if (key == ' ') {
            paused = !paused;
        } else if (key == 'p' || key == 'P') {
            next_seq = true;
            break;
        } else if (key == 'o' || key == 'O') {
            prev_seq = true;
            break;
        }

        if (paused) {
            if (key == 'm' || key == 'M') {
                if (frame_index + 1 < static_cast<int>(sequence_frames.size())) frame_index++;
            } else if (key == 'n' || key == 'N') {
                if (frame_index > 0) frame_index--;
            }
        } else {
            frame_index++;
        }
    }

    if (next_seq) {
        seq_index++;
        return true;
    } else if (prev_seq) {
        seq_index--;
        return true;
    }
    return false;
}

SequenceFeatures extractSequenceFeatures(const std::vector<cv::Mat>& mask_sequence, 
                                         const std::vector<cv::Rect>& bbox_sequence) {

    SequenceFeatures features;
    if (mask_sequence.size() < 2 || mask_sequence.size() != bbox_sequence.size()) {
        return features;
    }

    features.net_displacement = computeNetDisplacement(bbox_sequence);
    features.avg_velocity = computeAverageSequenceVelocity(bbox_sequence);
    features.stride_peaks = computeStrideFrequency_Width(bbox_sequence);

    return features;
}

StaticFeatures computeStaticFeatures(const std::vector<cv::Mat>& mask_sequence, 
                                     const std::vector<cv::Rect>& bbox_sequence) {

    StaticFeatures feats;
    if (mask_sequence.size() < 2 || bbox_sequence.empty()) {
        return feats;
    }

    std::vector<double> cy_values;        // Stores normalized vertical motion centroids (0.0 = top, 1.0 = bottom) for valid frames
    std::vector<double> x_offsets;        // Stores normalized lateral deviations from the bounding box midline (|cx - mid| / width)
    const size_t MIN_ACTIVE_PIXELS = 30; // Noise rejection threshold to skip nearly motionless frames

    for (size_t i = 1; i < mask_sequence.size(); ++i) {
        cv::Rect bbox = bbox_sequence[i];
        if (bbox.width <= 0 || bbox.height <= 0) continue;

        // Extract frame-to-frame motion mask
        cv::Mat diff;
        cv::absdiff(mask_sequence[i], mask_sequence[i - 1], diff);

        // Clip the bounding box to ensure it stays strictly within image boundaries
        cv::Rect img_bounds(0, 0, diff.cols, diff.rows);
        cv::Rect valid_roi = bbox & img_bounds;
        if (valid_roi.area() <= 0) continue;

        cv::Mat roi_diff = diff(valid_roi);

        // Collect coordinates of all pixels exhibiting temporal change
        std::vector<cv::Point> non_zero_pts;
        cv::findNonZero(roi_diff, non_zero_pts);

        // Discard frames with insufficient motion to prevent centroid noise
        if (non_zero_pts.size() >= MIN_ACTIVE_PIXELS) {
            double sum_x = 0.0;
            double sum_y = 0.0;

            for (const auto& pt : non_zero_pts) {
                sum_x += pt.x;
                sum_y += pt.y;
            }

            // Local motion center of mass within the bounding box
            double cx = sum_x / non_zero_pts.size();
            double cy = sum_y / non_zero_pts.size();

            // Normalize vertical reach by bbox height (0.0 = top of head, 1.0 = feet)
            double norm_cy = cy / bbox.height;
            cy_values.push_back(norm_cy);

            // Compute normalized lateral offset relative to the bbox's midline
            double center_x = bbox.width / 2.0;
            double norm_x_offset = std::abs(cx - center_x) / bbox.width;
            x_offsets.push_back(norm_x_offset);
        }
    }

    // Case where the sequence had no frames with sufficient motion
    if (cy_values.empty()) {
        return feats;
    }

    // Determine highest and lowest vertical reach across the entire action
    feats.min_cy = 1.0;
    feats.max_cy = 0.0;
    for (double cy : cy_values) {
        if (cy < feats.min_cy) feats.min_cy = cy;
        if (cy > feats.max_cy) feats.max_cy = cy;
    }
    feats.y_range = feats.max_cy - feats.min_cy;

    // Compute mean lateral asymmetry over all active frames
    double sum_x_offset = 0.0;
    for (double offset : x_offsets) {
        sum_x_offset += offset;
    }
    feats.avg_x_offset = sum_x_offset / x_offsets.size();
    feats.is_valid = true;

    return feats;
}

void printSequenceFeaturesTable(const std::string& seq_id, const SequenceFeatures& feats) {
    std::cout << std::left << std::setw(25) << seq_id 
              << " | Disp: " << std::setw(6) << std::fixed << std::setprecision(1) << feats.net_displacement
              << " | AvgVel: " << std::setw(5) << feats.avg_velocity
              << " | Peaks: " << std::setw(3) << feats.stride_peaks
              << std::endl;
}

void printStaticFeaturesTable(const std::string& seq_id, const StaticFeatures& feats) {
    std::cout << std::left << std::setw(25) << seq_id 
              << " | MinCY: " << std::setw(5) << std::fixed << std::setprecision(3) << feats.min_cy
              << " | YRange: " << std::setw(5) << feats.y_range
              << " | XOffset: " << std::setw(5) << feats.avg_x_offset
              << std::endl;
}

void printFinalReport(int total_processed, int total_correct,
                      const std::map<std::string, int>& class_total,
                      const std::map<std::string, int>& class_correct,
                      const std::vector<ErrorRecord>& error_log) {
                        
    std::cout << "\n\n==========================================" << std::endl;
    std::cout << "         FINAL EVALUATION SUMMARY         " << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << " Total sequences analyzed : " << total_processed << std::endl;
    std::cout << " Correctly classified     : " << total_correct << std::endl;
    
    float global_accuracy = (total_processed > 0) ? (static_cast<float>(total_correct) / total_processed) * 100.0f : 0.0f;
    std::cout << " Global Accuracy          : " << std::fixed << std::setprecision(2) << global_accuracy << "%" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    
    std::cout << " Per-Class Breakdown:" << std::endl;
    for (const auto& pair : class_total) {
        std::string cls = pair.first;
        int tot = pair.second;
        int cor = (class_correct.find(cls) != class_correct.end()) ? class_correct.at(cls) : 0;
        float acc = (tot > 0) ? (static_cast<float>(cor) / tot) * 100.0f : 0.0f;
        std::cout << "  - " << std::left << std::setw(15) << cls << ": " 
                  << cor << "/" << tot << " correct (" << acc << "%)" << std::endl;
    }
    
    std::cout << "------------------------------------------" << std::endl;
    std::cout << " Error Log (" << error_log.size() << " total errors):" << std::endl;
    if (error_log.empty()) {
        std::cout << "  No classification errors detected!" << std::endl;
    } else {
        std::map<std::string, int> confusion_matrix;
        for (const auto& err : error_log) {
            std::string key = "Ground Truth [" + err.real_action + "] mistaken for [" + err.predicted_action + "]";
            confusion_matrix[key]++;
        }

        for (const auto& conf : confusion_matrix) {
            std::cout << "  * " << conf.first << " -> " << conf.second << " time(s)" << std::endl;
        }
    }
    std::cout << "==========================================" << std::endl;
}