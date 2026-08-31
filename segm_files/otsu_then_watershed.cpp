#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "../tecniche_segmentazione.h" 

namespace fs = std::filesystem;

// Isola le variabili globali (usate dalla trackbar) per evitare conflitti
namespace {
    int dilation_val = 5; 
}

// Il vecchio main() diventa una funzione void
void test_otsu_then_watershed() {
    std::string folder_path = "./data"; 

    std::vector<cv::Mat> frames;
    std::vector<std::string> image_files;

    // Sostituito return -1 con return;
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) return; 

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) image_files.push_back(entry.path().string());
    }
    std::sort(image_files.begin(), image_files.end());
    
    // Sostituito return -1 con return;
    if (image_files.empty()) return; 

    for (const auto& file : image_files) {
        cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
        if (!img.empty()) frames.push_back(img);
    }

    std::string win_name = "Otsu-Guided Watershed: [Otsu Edges | Sure Regions | Watershed Result]";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name, 1800, 600);

    cv::createTrackbar("Edge Dilation (Close holes)", win_name, &dilation_val, 20);

    size_t i = 0;
    while (true) {
        const cv::Mat& frame = frames[i];

        // 1. Sfocatura e Sobel + Otsu
        cv::Mat blurred, grad_x, grad_y, sobel_mag, otsu_edges;
        cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);
        cv::Sobel(blurred, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(blurred, grad_y, CV_32F, 0, 1, 3);
        cv::magnitude(grad_x, grad_y, sobel_mag);
        cv::normalize(sobel_mag, sobel_mag, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::threshold(sobel_mag, otsu_edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // 2. Creazione della "Zona Sconosciuta" (Ispessiamo i bordi)
        int d_size = std::max(1, dilation_val);
        cv::Mat thick_edges;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(d_size, d_size));
        cv::dilate(otsu_edges, thick_edges, kernel);

        // 3. Generazione MARKER per il Watershed
        cv::Mat sure_regions;
        cv::bitwise_not(thick_edges, sure_regions); 

        cv::Mat markers;
        cv::connectedComponents(sure_regions, markers, 8, CV_32S);

        // 4. Esecuzione del WATERSHED
        cv::Mat frame_3ch;
        cv::cvtColor(blurred, frame_3ch, cv::COLOR_GRAY2BGR); 
        cv::watershed(frame_3ch, markers);

        // 5. Ricostruzione della Maschera Finale
        int bg_label = markers.at<int>(0, 0); 
        cv::Mat final_mask = cv::Mat::zeros(frame.size(), CV_8U);
        
        for (int r = 0; r < markers.rows; ++r) {
            for (int c = 0; c < markers.cols; ++c) {
                int label = markers.at<int>(r, c);
                if (label > 0 && label != bg_label) {
                    final_mask.at<uchar>(r, c) = 255;
                }
            }
        }

        // --- VISUALIZZAZIONE ---
        cv::Mat sure_vis;
        sure_regions.copyTo(sure_vis); 

        cv::Mat e_res, s_res, m_res;
        cv::resize(otsu_edges, e_res, cv::Size(600, 600), 0, 0, cv::INTER_NEAREST);
        cv::resize(sure_vis, s_res, cv::Size(600, 600), 0, 0, cv::INTER_NEAREST);
        cv::resize(final_mask, m_res, cv::Size(600, 600), 0, 0, cv::INTER_NEAREST);

        cv::Mat e_3ch, s_3ch, m_3ch, row, temp;
        cv::cvtColor(e_res, e_3ch, cv::COLOR_GRAY2BGR);
        cv::cvtColor(s_res, s_3ch, cv::COLOR_GRAY2BGR);
        cv::cvtColor(m_res, m_3ch, cv::COLOR_GRAY2BGR);

        cv::hconcat(e_3ch, s_3ch, temp);
        cv::hconcat(temp, m_3ch, row);

        cv::imshow(win_name, row);
        
        if ((char)cv::waitKey(100) == 'q') break;
        i = (i + 1) % frames.size();
    }

    cv::destroyAllWindows();
    // Rimosso return 0;
}