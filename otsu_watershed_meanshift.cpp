#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <filesystem> // Requires C++17
#include "tecniche_segmentazione.h" // Aggiunto per collegarlo al progetto

namespace fs = std::filesystem;

// Il vecchio main() ora è una funzione richiamabile
void comparison_otsu_watershed_meanshift() {
    // Path to the folder containing the frames
    std::string folder_path = "./data"; 

    std::vector<cv::Mat> frames;
    std::vector<std::string> image_files;

    // 1. Read all image paths from the folder
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        std::cerr << "[ERROR] Folder does not exist: " << folder_path << std::endl;
        return; // Sostituito return -1 con return
    }

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp") {
                image_files.push_back(entry.path().string());
            }
        }
    }

    // Sort files to keep the correct chronological sequence of frames
    std::sort(image_files.begin(), image_files.end());

    if (image_files.empty()) {
        std::cerr << "[ERROR] No image frames found in the folder!" << std::endl;
        return; // Sostituito return -1 con return
    }

    std::cout << "[INFO] Found " << image_files.size() << " frames. Loading..." << std::endl;

    // 2. Load frames in grayscale
    for (const auto& file : image_files) {
        cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
        if (!img.empty()) {
            frames.push_back(img);
        }
    }

    std::string win_name = "Comparison: [Original | Bilateral+Otsu | Watershed | Mean Shift]";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    // Impostiamo una finestra più larga per contenere 4 pannelli affiancati (es. 2400x600)
    cv::resizeWindow(win_name, 2400, 600);

    std::cout << "\n[INFO] Playing 4-way comparison in loop. Press 'q' or 'Q' to exit.\n" << std::endl;

    // 3. Loop continuously through frames until 'q' is pressed
    size_t i = 0;
    while (true) {
        const cv::Mat& frame = frames[i];

        // --- METHOD 1: Bilateral + Otsu ---
        cv::Mat filtered_otsu, otsu_frame;
        cv::bilateralFilter(frame, filtered_otsu, 9, 75, 75);
        cv::threshold(filtered_otsu, otsu_frame, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // --- METHOD 2: Bilateral + Sobel + Watershed ---
        cv::Mat filtered_ws;
        cv::bilateralFilter(frame, filtered_ws, 9, 75, 75);

        cv::Mat thresh;
        cv::threshold(filtered_ws, thresh, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

        cv::Mat markers;
        cv::connectedComponents(thresh, markers);
        markers = markers + 1; 

        cv::Mat frame_3ch_ws;
        cv::cvtColor(frame, frame_3ch_ws, cv::COLOR_GRAY2BGR);
        cv::watershed(frame_3ch_ws, markers);

        cv::Mat watershed_result = cv::Mat::zeros(frame.size(), CV_8U);
        for (int r = 0; r < markers.rows; r++) {
            for (int c = 0; c < markers.cols; c++) {
                int index = markers.at<int>(r, c);
                if (index > 1) {
                    watershed_result.at<uchar>(r, c) = 255; 
                } else {
                    watershed_result.at<uchar>(r, c) = 0;   
                }
            }
        }

        // --- METHOD 3: Bilateral + Mean Shift ---
        cv::Mat filtered_ms, ms_bgr, ms_result;
        cv::bilateralFilter(frame, filtered_ms, 9, 75, 75);
        
        // Mean Shift expects a 3-channel image
        cv::cvtColor(filtered_ms, ms_bgr, cv::COLOR_GRAY2BGR);
        
        // Apply pyrMeanShiftFiltering (spatial window = 20, color window = 40 as tuning examples)
        cv::pyrMeanShiftFiltering(ms_bgr, ms_bgr, 20, 40);
        
        // Convert back to grayscale for uniform comparison format
        cv::cvtColor(ms_bgr, ms_result, cv::COLOR_BGR2GRAY);

        // --- RESIZE FOR LARGER VIEW (600x600 per panel) ---
        int target_height = 400;
        int target_width = 400;
        
        cv::Mat frame_resized, otsu_resized, ws_resized, ms_resized;
        cv::resize(frame, frame_resized, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
        cv::resize(otsu_frame, otsu_resized, cv::Size(target_width, target_height), 0, 0, cv::INTER_NEAREST);
        cv::resize(watershed_result, ws_resized, cv::Size(target_width, target_height), 0, 0, cv::INTER_NEAREST);
        cv::resize(ms_result, ms_resized, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);

        // --- SIDE-BY-SIDE CONCATENATION (4 panels) ---
        cv::Mat orig_3ch, otsu_3ch, ws_3ch, ms_3ch;
        if (frame_resized.channels() == 1) cv::cvtColor(frame_resized, orig_3ch, cv::COLOR_GRAY2BGR);
        else orig_3ch = frame_resized;

        if (otsu_resized.channels() == 1) cv::cvtColor(otsu_resized, otsu_3ch, cv::COLOR_GRAY2BGR);
        else otsu_3ch = otsu_resized;

        if (ws_resized.channels() == 1) cv::cvtColor(ws_resized, ws_3ch, cv::COLOR_GRAY2BGR);
        else ws_3ch = ws_resized;

        if (ms_resized.channels() == 1) cv::cvtColor(ms_resized, ms_3ch, cv::COLOR_GRAY2BGR);
        else ms_3ch = ms_resized;

        cv::Mat row1, row2, final_comparison;
        cv::hconcat(orig_3ch, otsu_3ch, row1);
        cv::hconcat(ws_3ch, ms_3ch, row2);
        
        // Se preferisci tutto su una singola riga orizzontale a 4 colonne:
        cv::Mat temp_hcat;
        cv::hconcat(orig_3ch, otsu_3ch, temp_hcat);
        cv::hconcat(temp_hcat, ws_3ch, temp_hcat);
        cv::hconcat(temp_hcat, ms_3ch, final_comparison);

        // Show the comparison window
        cv::imshow(win_name, final_comparison);

        // Wait ~100ms per frame, check if 'q' or 'Q' is pressed
        char key = (char)cv::waitKey(100);
        if (key == 'q' || key == 'Q') {
            break;
        }

        // Advance frame index, loop back to 0 at the end
        i = (i + 1) % frames.size();
    }

    cv::destroyAllWindows();
    std::cout << "\n=== TEST TERMINATO ===" << std::endl;
}