/*
 * ==============================================================================
 * WATERSHED SEGMENTATION PIPELINE - PROJECT NOTES & EVOLUTION
 * ==============================================================================
 * How i reached this solution:
 * 1. Initial approach: i started with a basic global Otsu thresholding combined with 
 *    Gaussian background subtraction. While it worked well for uniform studio walls, 
 *    it completely broke down on outdoor sequences (e.g., grass/sky) and zoomed-out frames.
 * 2. Fixing binarization: i replaced the global background subtraction with 
 *    'cv::adaptiveThreshold', which calculates local thresholds block-by-block. This 
 *    made the pipeline robust across different lighting conditions and backgrounds.
 * 3. Handling legs & noise: to catch the legs without triggering floor noise, i 
 *    added a mild vertical morphological closing followed by an opening step. I then 
 *    filtered contours by area (dropping tiny blobs) to handle zooming and eliminate 
 *    spurious artifacts uniformly across all sequences without hardcoded coordinates.
 * 4. Automation: Finally, i implemented a recursive directory iterator to traverse 
 *    the nested dataset structure (Actions -> Sequences -> data -> frames) automatically.
 *
 * Algorithm overview:
 * - Bilateral Filtering: Smooths flat regions while keeping object edges sharp.
 * - Adaptive Thresholding: Binarizes the image locally to isolate the dark subject.
 * - Morphology: Uses small vertical kernels to bridge gaps (legs) and clean noise.
 * - Contour Area Filtering: Keeps only sufficiently large blobs, discarding small noise.
 * - Marker Generation: Sets image borders as a safe background (1) and filtered blobs 
 *   as foreground seeds (2), leaving unknown pixels as (0).
 * - Sobel Gradient & Watershed: Computes edge gradients as barriers and runs OpenCV 
 *   watershed to flood-fill and segment the person precisely.
 * ==============================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "../tecniche_segmentazione.h" 

namespace fs = std::filesystem;

void test3_watershed() {
    // Automatically locate the root Sequences folder relative to the build directory
    std::string root_folder = "../data/Sequences"; 

    if (!fs::exists(root_folder)) {
        std::cerr << "[ERROR] Could not find folder: " << root_folder << std::endl;
        return;
    }

    struct SequenceData {
        std::string name;
        std::vector<cv::Mat> frames;
    };
    std::vector<SequenceData> all_sequences;

    // Traverse action folders (e.g., boxing, waving...)
    for (const auto& action_entry : fs::directory_iterator(root_folder)) {
        if (!action_entry.is_directory()) continue;
        
        std::string action_name = action_entry.path().filename().string();

        // Traverse sequence subfolders (e.g., person02_boxing_d4...)
        for (const auto& sub_entry : fs::directory_iterator(action_entry.path())) {
            if (!sub_entry.is_directory()) continue;

            // Enter the inner "data" folder containing the actual frames
            fs::path target_data_path = sub_entry.path() / "data";
            if (!fs::exists(target_data_path) || !fs::is_directory(target_data_path)) continue;

            std::vector<std::string> image_files;
            for (const auto& file_entry : fs::directory_iterator(target_data_path)) {
                if (file_entry.is_regular_file()) {
                    image_files.push_back(file_entry.path().string());
                }
            }
            std::sort(image_files.begin(), image_files.end());

            if (!image_files.empty()) {
                SequenceData seq;
                seq.name = action_name + "/" + sub_entry.path().filename().string();
                for (const auto& file : image_files) {
                    cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
                    if (!img.empty()) seq.frames.push_back(img);
                }
                if (!seq.frames.empty()) {
                    all_sequences.push_back(seq);
                }
            }
        }
    }

    if (all_sequences.empty()) {
        std::cout << "[WARNING] No sequences found in the specified directory!" << std::endl;
        return;
    }

    std::cout << "\n[INFO] Found " << all_sequences.size() << " sequences in total." << std::endl;
    std::cout << "[INFO] Playing sequences. Press 'SPACE' to skip to the next sequence or 'q' to exit.\n" << std::endl;

    // Scale factor for on-screen display magnification
    double display_scale = 3.0; // Increased to 3.0 to further enlarge images on screen
    std::string win_name = "Watershed Pipeline Sequence: [Original | Bilateral | Markers | Gradient | Watershed]";
    
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    // Dynamically adjust initial window size based on the enlarged resolution
    // Assuming a base frame resolution of about 160x120 multiplied by 5 display panels
    cv::resizeWindow(win_name, 160 * 5 * display_scale, 120 * display_scale);

    // Loop through all sequences
    size_t seq_idx = 0;
    while (seq_idx < all_sequences.size()) {
        const auto& current_seq = all_sequences[seq_idx];
        std::cout << "[PLAYING] Sequence (" << seq_idx + 1 << "/" << all_sequences.size() << "): " << current_seq.name << std::endl;

        size_t i = 0;
        bool change_sequence = false;

        while (!change_sequence) {
            const cv::Mat& frame = current_seq.frames[i];

            // 1. Bilateral filter to smooth noise while keeping edges sharp
            cv::Mat bilateral_filtered;
            cv::bilateralFilter(frame, bilateral_filtered, 10, 20.0, 20.0);

            // 2. Adaptive thresholding for robust binarization across different backgrounds/zoom
            cv::Mat otsu_mask;
            cv::adaptiveThreshold(bilateral_filtered, otsu_mask, 255, 
                                  cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv::THRESH_BINARY_INV, 11, 4);

            // Short vertical closing to connect torso and legs without hitting the floor
            cv::Mat vertical_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 5));
            cv::morphologyEx(otsu_mask, otsu_mask, cv::MORPH_CLOSE, vertical_kernel);

            // Opening to break unwanted horizontal connections/noise
            cv::Mat cleaner_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::morphologyEx(otsu_mask, otsu_mask, cv::MORPH_OPEN, cleaner_kernel);

            // Final cleanup using contour areas (area > 100 handles zoom-outs safely)
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(otsu_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            cv::Mat cleaned_otsu = cv::Mat::zeros(otsu_mask.size(), CV_8U);
            for (size_t k = 0; k < contours.size(); k++) {
                double area = cv::contourArea(contours[k]);
                if (area > 100) { 
                    cv::drawContours(cleaned_otsu, contours, (int)k, cv::Scalar(255), cv::FILLED);
                }
            }

            // 3. Create markers: fixed background borders (1) + foreground seeds (2)
            cv::Mat markers = cv::Mat::zeros(frame.size(), CV_32S);
            int border_thickness = 15; 
            cv::rectangle(markers, cv::Rect(0, 0, markers.cols, markers.rows), cv::Scalar(1), border_thickness);

            for (int r = 0; r < markers.rows; r++) {
                for (int c = 0; c < markers.cols; c++) {
                    if (markers.at<int>(r, c) == 0 && cleaned_otsu.at<uchar>(r, c) == 255) {
                        markers.at<int>(r, c) = 2; 
                    }
                }
            }

            // Visualize initial foreground markers in blue
            cv::Mat marker_visual;
            cv::cvtColor(bilateral_filtered, marker_visual, cv::COLOR_GRAY2BGR);
            for (int r = 0; r < markers.rows; r++) {
                for (int c = 0; c < markers.cols; c++) {
                    if (markers.at<int>(r, c) == 2) {
                        marker_visual.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 0, 0); 
                    }
                }
            }

            // 4. Gradient calculation using Sobel on the filtered image
            cv::Mat grad_x, grad_y, gradient_img;
            cv::Sobel(bilateral_filtered, grad_x, CV_16S, 1, 0, 3);
            cv::Sobel(bilateral_filtered, grad_y, CV_16S, 0, 1, 3);
            cv::Mat abs_grad_x, abs_grad_y;
            cv::convertScaleAbs(grad_x, abs_grad_x);
            cv::convertScaleAbs(grad_y, abs_grad_y);
            cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradient_img);

            // 5. Watershed segmentation
            cv::Mat gradient_bgr;
            cv::cvtColor(gradient_img, gradient_bgr, cv::COLOR_GRAY2BGR);
            cv::watershed(gradient_bgr, markers);

            cv::Mat watershed_mask = cv::Mat::zeros(frame.size(), CV_8U);
            for (int r = 0; r < markers.rows; r++) {
                for (int c = 0; c < markers.cols; c++) {
                    if (markers.at<int>(r, c) == 2) {
                        watershed_mask.at<uchar>(r, c) = 255;
                    }
                }
            }

            // --- Resize and layout composition for display (using display_scale) ---
            cv::Mat r_orig, r_bilat, r_markers, r_grad, r_ws;
            cv::resize(frame, r_orig, cv::Size(), display_scale, display_scale, cv::INTER_NEAREST);
            cv::resize(bilateral_filtered, r_bilat, cv::Size(), display_scale, display_scale, cv::INTER_NEAREST);
            cv::resize(marker_visual, r_markers, cv::Size(), display_scale, display_scale, cv::INTER_NEAREST);
            cv::resize(gradient_img, r_grad, cv::Size(), display_scale, display_scale, cv::INTER_NEAREST);
            cv::resize(watershed_mask, r_ws, cv::Size(), display_scale, display_scale, cv::INTER_NEAREST);

            cv::Mat b_orig, b_bilat, b_grad, b_ws;
            cv::cvtColor(r_orig, b_orig, cv::COLOR_GRAY2BGR);
            cv::cvtColor(r_bilat, b_bilat, cv::COLOR_GRAY2BGR);
            cv::cvtColor(r_grad, b_grad, cv::COLOR_GRAY2BGR);
            cv::cvtColor(r_ws, b_ws, cv::COLOR_GRAY2BGR);

            cv::Mat temp1, temp2, temp3, final_display;
            cv::hconcat(b_orig, b_bilat, temp1);
            cv::hconcat(temp1, r_markers, temp2);
            cv::hconcat(temp2, b_grad, temp3);
            cv::hconcat(temp3, b_ws, final_display);

            cv::imshow(win_name, final_display);
            
            char key = (char)cv::waitKey(40);
            if (key == 'q' || key == 'Q') {
                cv::destroyAllWindows();
                return;
            } else if (key == ' ') { // Press Space to skip to the next sequence
                change_sequence = true;
            }

            i = (i + 1) % current_seq.frames.size();
            if (i == 0) {
                change_sequence = true; // Automatically proceed to the next sequence at the end
            }
        }
        seq_idx++;
    }

    cv::destroyAllWindows();
}