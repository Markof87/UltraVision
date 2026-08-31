#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "../tecniche_segmentazione.h" 

namespace fs = std::filesystem;

// Isola le variabili globali della trackbar
namespace {
    int slider_morph_grid = 1; 
    int slider_split_y = 50; // Posizione del taglio orizzontale in percentuale (0-100)
}

void test_otsu_2reg_dynamic() {
    std::string folder_path = "./data"; 

    std::vector<std::string> image_files;

    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) return; 

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp") {
                image_files.push_back(entry.path().string());
            }
        }
    }
    std::sort(image_files.begin(), image_files.end());

    if (image_files.empty()) return; 

    std::vector<cv::Mat> frames;
    std::cout << "[INFO] Loading " << image_files.size() << " frames..." << std::endl;
    for (const auto& file : image_files) {
        cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
        if (!img.empty()) frames.push_back(img);
    }

    std::string win_name = "Otsu Top/Bottom: [Original | Raw Otsu | Cleaned Mask]";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name, 1800, 600);
    cv::moveWindow(win_name, 100, 100);

    // Trackbars per Kernel e per la posizione della linea
    cv::createTrackbar("Kernel Size", win_name, &slider_morph_grid, 5, nullptr);
    cv::createTrackbar("Split Y (%)", win_name, &slider_split_y, 100, nullptr);

    std::cout << "\n[INFO] Playing loop. Tune fine kernel size and split position. Press 'q' to exit.\n" << std::endl;

    size_t i = 0;
    while (true) {
        const cv::Mat& frame = frames[i];
        int rows = frame.rows;
        int cols = frame.cols;

        // Sicurezza: blocca lo split tra 1% e 99% per evitare rettangoli con altezza 0
        int safe_split = std::max(1, std::min(slider_split_y, 99));
        
        // Calcola l'altezza in base alla percentuale dello slider
        int mid_h = (rows * safe_split) / 100;

        // 1. APPLICAZIONE DI OTSU SU 2 REGIONI (Sopra e Sotto)
        cv::Mat grid_otsu_res = cv::Mat::zeros(rows, cols, CV_8U);

        cv::Rect regions[2] = {
            cv::Rect(0, 0, cols, mid_h),              // Metà superiore
            cv::Rect(0, mid_h, cols, rows - mid_h)    // Metà inferiore
        };

        for (int q = 0; q < 2; ++q) {
            cv::Mat sub_img = frame(regions[q]);
            cv::Mat sub_thresh;
            cv::threshold(sub_img, sub_thresh, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
            sub_thresh.copyTo(grid_otsu_res(regions[q]));
        }

        // 2. PULIZIA MORFOLOGICA 
        int ksize = slider_morph_grid; 
        cv::Mat cleaned = grid_otsu_res.clone();

        if (ksize > 0) {
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));
            cv::morphologyEx(grid_otsu_res, cleaned, cv::MORPH_OPEN, kernel);
        }

        // --- VISUALIZZAZIONE ---
        cv::Mat f_res, r_res, c_res;
        cv::resize(frame, f_res, cv::Size(600, 600), 0, 0, cv::INTER_LINEAR);
        cv::resize(grid_otsu_res, r_res, cv::Size(600, 600), 0, 0, cv::INTER_NEAREST); 
        cv::resize(cleaned, c_res, cv::Size(600, 600), 0, 0, cv::INTER_NEAREST);      

        cv::Mat f_3ch, r_3ch, c_3ch, row, temp;
        cv::cvtColor(f_res, f_3ch, cv::COLOR_GRAY2BGR);
        cv::cvtColor(r_res, r_3ch, cv::COLOR_GRAY2BGR);
        cv::cvtColor(c_res, c_3ch, cv::COLOR_GRAY2BGR);

        // Disegno la linea divisoria calcolandone l'altezza sulla finestra 600x600
        int line_y = (600 * safe_split) / 100;
        cv::line(r_3ch, cv::Point(0, line_y), cv::Point(600, line_y), cv::Scalar(0, 255, 0), 1);

        std::string info = "Kernel: " + std::to_string(ksize) + "x" + std::to_string(ksize) + " | Split: " + std::to_string(safe_split) + "%";
        cv::putText(f_3ch, info, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        cv::hconcat(f_3ch, r_3ch, temp);
        cv::hconcat(temp, c_3ch, row);

        cv::imshow(win_name, row);
        
        char key = (char)cv::waitKey(100);
        if (key == 'q' || key == 'Q') break;

        i = (i + 1) % frames.size();
    }

    cv::destroyAllWindows();
}