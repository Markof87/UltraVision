#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "DatasetManager.hpp"
#include "Segmentation.hpp"

int main() {
    std::string sequences_path = "../Sequences";

    // Phase 1: Load dataset
    std::cout << "[INFO] Starting sequence loading..." << std::endl;
    std::vector<SequenceData> dataset = load_sequences(sequences_path);

    if (dataset.empty()) {
        std::cerr << "[ERROR] No sequences found! Check the path." << std::endl;
        return -1;
    }

    std::cout << "\n[INFO] Starting Phase 2: Processing entire dataset with multiple segmentation methods..." << std::endl;

    std::vector<SequenceData> compared_dataset;
    compared_dataset.reserve(dataset.size());

    // Process every sequence in the dataset
    for (const auto& seq : dataset) {
        // 1. Apply both segmentation techniques
        std::vector<cv::Mat> otsu_frames = apply_bilateral_and_otsu(seq.frames);
        std::vector<cv::Mat> watershed_frames = apply_bilateral_sobel_watershed(seq.frames);

        std::vector<cv::Mat> combined_frames;
        combined_frames.reserve(seq.frames.size());

        // 2. Combine frames side-by-side: [Original | Otsu | Watershed]
        for (size_t i = 0; i < seq.frames.size(); ++i) {
            cv::Mat orig = seq.frames[i];
            cv::Mat otsu = otsu_frames[i];
            cv::Mat watershed = watershed_frames[i];

            // Ensure all images have 3 channels (BGR) so hconcat works seamlessly
            cv::Mat orig_3ch, otsu_3ch, watershed_3ch;
            
            if (orig.channels() == 1) cv::cvtColor(orig, orig_3ch, cv::COLOR_GRAY2BGR);
            else orig_3ch = orig;

            if (otsu.channels() == 1) cv::cvtColor(otsu, otsu_3ch, cv::COLOR_GRAY2BGR);
            else otsu_3ch = otsu;

            if (watershed.channels() == 1) cv::cvtColor(watershed, watershed_3ch, cv::COLOR_GRAY2BGR);
            else watershed_3ch = watershed;

            // Concatenate horizontally: [Original | Otsu | Watershed] in a single row
            cv::Mat comparison_row;
            cv::Mat temp_hcat;
            cv::hconcat(orig_3ch, otsu_3ch, temp_hcat);
            cv::hconcat(temp_hcat, watershed_3ch, comparison_row);

            combined_frames.push_back(comparison_row);
        }

        std::string new_title = "Comparison (Orig | Otsu | Watershed) - " + seq.complete_name;

        compared_dataset.push_back({
            combined_frames, 
            seq.action, 
            seq.person, 
            seq.env, 
            new_title
        });
    }

    std::cout << "[INFO] Dataset processing completed successfully!" << std::endl;

    // 3. Call the visualization menu to let the user choose how to display the comparison dataset
    handle_visualization_menu(compared_dataset);

    std::cout << "\n=== PROGRAM TERMINATED SUCCESSFULLY ===" << std::endl;

    return 0;
}