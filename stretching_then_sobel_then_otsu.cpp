#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "tecniche_segmentazione.h" // Aggiunto per il progetto modulare

namespace fs = std::filesystem;

// Isola le variabili globali nel namespace anonimo per evitare conflitti con altri file
namespace {
    int r1_val = 80;
    int s1_val = 0;
    int r2_val = 150;
    int s2_val = 255;
}

// Il main diventa la nostra funzione di test

void test_stretching_then_sobel_then_otsu() {
    std::string folder_path = "./data"; 

    std::vector<cv::Mat> frames;
    std::vector<std::string> image_files;

    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        std::cerr << "[ERROR] Cartella non trovata: " << folder_path << std::endl;
        return; // Sostituito return -1 con return;
    }

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp") {
                image_files.push_back(entry.path().string());
            }
        }
    }
    std::sort(image_files.begin(), image_files.end());

    if (image_files.empty()) return; // Sostituito return -1 con return;

    std::cout << "[INFO] Caricamento di " << image_files.size() << " frame..." << std::endl;
    for (const auto& file : image_files) {
        cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
        if (!img.empty()) frames.push_back(img);
    }

    std::string win_name = "Pre-Sobel Stretching Comparison";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name, 1500, 1000);

    // Trackbars per regolare la curva sull'immagine originale
    cv::createTrackbar("r1 (Input Dark)", win_name, &r1_val, 255);
    cv::createTrackbar("s1 (Output Dark)", win_name, &s1_val, 255);
    cv::createTrackbar("r2 (Input Bright)", win_name, &r2_val, 255);
    cv::createTrackbar("s2 (Output Bright)", win_name, &s2_val, 255);

    size_t i = 0;
    while (true) {
        const cv::Mat& frame = frames[i];

        // 1. Sfocatura per attenuare il rumore prima di ogni operazione
        cv::Mat blurred;
        cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);

        // =================================================================
        // PIPELINE 1: NORMALE (Senza Stretching)
        // =================================================================
        cv::Mat grad_x_norm, grad_y_norm, sobel_normal, otsu_normal;
        
        cv::Sobel(blurred, grad_x_norm, CV_32F, 1, 0, 3);
        cv::Sobel(blurred, grad_y_norm, CV_32F, 0, 1, 3);
        cv::magnitude(grad_x_norm, grad_y_norm, sobel_normal);
        cv::normalize(sobel_normal, sobel_normal, 0, 255, cv::NORM_MINMAX, CV_8U);
        
        cv::threshold(sobel_normal, otsu_normal, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // =================================================================
        // PIPELINE 2: STRETCHING APPLICATO PRIMA DI SOBEL
        // =================================================================
        int curr_r1 = std::max(0, std::min(r1_val, r2_val));
        int curr_r2 = std::max(curr_r1, std::min(r2_val, 255));
        int curr_s1 = std::max(0, std::min(s1_val, 255));
        int curr_s2 = std::max(0, std::min(s2_val, 255));

        // Creazione LUT
        cv::Mat lut(1, 256, CV_8U);
        uchar* p_lut = lut.data;
        for (int r = 0; r < 256; r++) {
            double s = 0.0;
            if (r < curr_r1) {
                s = (curr_r1 > 0) ? ((double)curr_s1 / curr_r1) * r : 0;
            } else if (r >= curr_r1 && r <= curr_r2) {
                s = (curr_r2 != curr_r1) ? curr_s1 + ((double)(curr_s2 - curr_s1) / (curr_r2 - curr_r1)) * (r - curr_r1) : curr_s1;
            } else {
                s = (255 != curr_r2) ? curr_s2 + ((double)(255 - curr_s2) / (255 - curr_r2)) * (r - curr_r2) : curr_s2;
            }
            p_lut[r] = cv::saturate_cast<uchar>(s);
        }

        // 2A. Applichiamo lo stretching all'immagine originale (sfocata)
        cv::Mat stretched_frame;
        cv::LUT(blurred, lut, stretched_frame);

        // 2B. Calcoliamo Sobel SULL'IMMAGINE STRETCCIATA
        cv::Mat grad_x_str, grad_y_str, sobel_stretched, otsu_stretched;
        cv::Sobel(stretched_frame, grad_x_str, CV_32F, 1, 0, 3);
        cv::Sobel(stretched_frame, grad_y_str, CV_32F, 0, 1, 3);
        cv::magnitude(grad_x_str, grad_y_str, sobel_stretched);
        cv::normalize(sobel_stretched, sobel_stretched, 0, 255, cv::NORM_MINMAX, CV_8U);

        // 2C. Otsu sui nuovi gradienti
        cv::threshold(sobel_stretched, otsu_stretched, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // =================================================================
        // 3. COSTRUZIONE GRIGLIA VISIVA (2x3)
        // =================================================================
        int sz = 500; // Dimensione dei singoli pannelli
        cv::Mat f_res, sn_res, on_res, sf_res, ss_res, os_res;
        
        // Riga 1: Originale | Sobel Normale | Otsu Normale
        cv::resize(frame, f_res, cv::Size(sz, sz), 0, 0, cv::INTER_LINEAR);
        cv::resize(sobel_normal, sn_res, cv::Size(sz, sz), 0, 0, cv::INTER_LINEAR);
        cv::resize(otsu_normal, on_res, cv::Size(sz, sz), 0, 0, cv::INTER_NEAREST);
        
        // Riga 2: Originale Stretcciato | Sobel Stretcciato | Otsu Stretcciato
        cv::resize(stretched_frame, sf_res, cv::Size(sz, sz), 0, 0, cv::INTER_LINEAR);
        cv::resize(sobel_stretched, ss_res, cv::Size(sz, sz), 0, 0, cv::INTER_LINEAR);
        cv::resize(otsu_stretched, os_res, cv::Size(sz, sz), 0, 0, cv::INTER_NEAREST);

        // Conversione a 3 canali
        cv::cvtColor(f_res, f_res, cv::COLOR_GRAY2BGR);
        cv::cvtColor(sn_res, sn_res, cv::COLOR_GRAY2BGR);
        cv::cvtColor(on_res, on_res, cv::COLOR_GRAY2BGR);
        cv::cvtColor(sf_res, sf_res, cv::COLOR_GRAY2BGR);
        cv::cvtColor(ss_res, ss_res, cv::COLOR_GRAY2BGR);
        cv::cvtColor(os_res, os_res, cv::COLOR_GRAY2BGR);

        // Composizione
        cv::Mat row1, row2, grid;
        std::vector<cv::Mat> images_row1 = {f_res, sn_res, on_res};
        std::vector<cv::Mat> images_row2 = {sf_res, ss_res, os_res};
        
        cv::hconcat(images_row1, row1);
        cv::hconcat(images_row2, row2);
        cv::vconcat(row1, row2, grid);

        cv::imshow(win_name, grid);
        
        char key = (char)cv::waitKey(100);    
        if (key == 'q' || key == 'Q') break;

        i = (i + 1) % frames.size();
    }

    cv::destroyAllWindows();
    // Rimosso return 0;
}