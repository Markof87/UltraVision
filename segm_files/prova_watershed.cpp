#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "../tecniche_segmentazione.h" 

namespace fs = std::filesystem;

void test2_watershed() {
    std::string folder_path = "./data"; 
    std::vector<cv::Mat> frames;
    std::vector<std::string> image_files;

    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) return; 

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) image_files.push_back(entry.path().string());
    }
    std::sort(image_files.begin(), image_files.end());
    
    if (image_files.empty()) return; 

    for (const auto& file : image_files) {
        cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
        if (!img.empty()) frames.push_back(img);
    }

    double scale_factor = 1.5; 
    std::string win_name = "Watershed Pipeline Sequence: [Original | Bilateral | Markers | Gradient | Watershed]";
    
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name, 1500, 400);

    std::cout << "\n[INFO] Riproduzione in loop avviata. Premi 'q' per uscire.\n" << std::endl;

    size_t i = 0;
    while (true) {
        const cv::Mat& frame = frames[i];

       // 1. Bilateral Filter
        cv::Mat bilateral_filtered;
        cv::bilateralFilter(frame, bilateral_filtered, 10, 20.0, 20.0);

        // Stima dello sfondo globale
        cv::Mat bg_estimate, enhanced_img;
        cv::GaussianBlur(bilateral_filtered, bg_estimate, cv::Size(101, 101), 0);
        cv::subtract(bg_estimate, bilateral_filtered, enhanced_img);

        // 2. Otsu + Chiusura verticale per prendere le gambe
        cv::Mat otsu_mask;
        cv::threshold(enhanced_img, otsu_mask, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

        cv::Mat vertical_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 15));
        cv::morphologyEx(otsu_mask, otsu_mask, cv::MORPH_CLOSE, vertical_kernel);

        // --- APERTURA CON KERNEL VERTICALE (Cancella i residui orizzontali/orizzontalmente estesi) ---
        // Invece di un kernel quadrato, usiamo un kernel sviluppato in verticale per ripulire
        cv::Mat cleaner_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 9));
        cv::morphologyEx(otsu_mask, otsu_mask, cv::MORPH_OPEN, cleaner_kernel);

        // --- PULIZIA SOLO TRAMITE AREA DEI CONTORNI (Senza Bounding Box) ---
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(otsu_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat cleaned_otsu = cv::Mat::zeros(otsu_mask.size(), CV_8U);
        for (size_t k = 0; k < contours.size(); k++) {
            double area = cv::contourArea(contours[k]);
            
            // Filtriamo solo in base all'area minima, lasciando fare la selezione della forma ai kernel morfologici
            if (area > 500) {
                cv::drawContours(cleaned_otsu, contours, (int)k, cv::Scalar(255), cv::FILLED);
            }
        }

        // 3. Creazione Markers: Solo bordi esterni protettivi + Otsu pulito
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

        // Visualizzazione dei marker
        cv::Mat marker_visual;
        cv::cvtColor(bilateral_filtered, marker_visual, cv::COLOR_GRAY2BGR);
        for (int r = 0; r < markers.rows; r++) {
            for (int c = 0; c < markers.cols; c++) {
                if (markers.at<int>(r, c) == 2) {
                    marker_visual.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 0, 0); // Blu
                }
            }
        }

        // 4. Gradient (Sobel on Bilateral)
        cv::Mat grad_x, grad_y, gradient_img;
        cv::Sobel(bilateral_filtered, grad_x, CV_16S, 1, 0, 3);
        cv::Sobel(bilateral_filtered, grad_y, CV_16S, 0, 1, 3);
        cv::Mat abs_grad_x, abs_grad_y;
        cv::convertScaleAbs(grad_x, abs_grad_x);
        cv::convertScaleAbs(grad_y, abs_grad_y);
        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradient_img);

        // 5. Watershed
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

        // --- Ridimensionamento e conversione dei pannelli ---
        cv::Mat r_orig, r_bilat, r_markers, r_grad, r_ws;
        cv::resize(frame, r_orig, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
        cv::resize(bilateral_filtered, r_bilat, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
        cv::resize(marker_visual, r_markers, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
        cv::resize(gradient_img, r_grad, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
        cv::resize(watershed_mask, r_ws, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);

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
            break;
        }

        i = (i + 1) % frames.size();
    }

    cv::destroyAllWindows();
}