#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "../tecniche_segmentazione.h" 

namespace fs = std::filesystem;

// Il vecchio main() diventa una funzione void
void test_sobel_then_otsu() {
    std::string folder_path = "./data"; 

    std::vector<cv::Mat> frames;
    std::vector<std::string> image_files;

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
    std::sort(image_files.begin(), image_files.end());

    if (image_files.empty()) return; // Sostituito return -1 con return

    std::cout << "[INFO] Loading " << image_files.size() << " frames..." << std::endl;
    for (const auto& file : image_files) {
        cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
        if (!img.empty()) frames.push_back(img);
    }

    std::string win_name = "Edge Detection: [Original | Sobel Magnitude | Sobel + Otsu]";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(win_name, 1800, 600);

    std::cout << "\n[INFO] Playing loop. Press 'q' or 'Q' to exit.\n" << std::endl;

    size_t i = 0;
    while (true) {
        const cv::Mat& frame = frames[i];

        // 1. Sfocatura per mitigare il rumore di fondo (texture erba)
        cv::Mat blurred;
        cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);

        // 2. Estrazione Bordi (Sobel Magnitude) -> SECONDA IMMAGINE
        cv::Mat grad_x, grad_y, grad_mag;
        cv::Sobel(blurred, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(blurred, grad_y, CV_32F, 0, 1, 3);
        cv::magnitude(grad_x, grad_y, grad_mag);
        cv::normalize(grad_mag, grad_mag, 0, 255, cv::NORM_MINMAX, CV_8U);

        // 3. Binarizzazione di Otsu sui bordi -> TERZA IMMAGINE
        cv::Mat edges;
        cv::threshold(grad_mag, edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // --- VISUALIZZAZIONE ---
        cv::Mat f_res, s_res, e_res;
        cv::resize(frame, f_res, cv::Size(600, 600), 0, 0, cv::INTER_LINEAR);
        cv::resize(grad_mag, s_res, cv::Size(600, 600), 0, 0, cv::INTER_LINEAR);
        cv::resize(edges, e_res, cv::Size(600, 600), 0, 0, cv::INTER_NEAREST);

        cv::Mat f_3ch, s_3ch, e_3ch, row, temp;
        cv::cvtColor(f_res, f_3ch, cv::COLOR_GRAY2BGR);
        cv::cvtColor(s_res, s_3ch, cv::COLOR_GRAY2BGR);
        cv::cvtColor(e_res, e_3ch, cv::COLOR_GRAY2BGR);

        cv::hconcat(f_3ch, s_3ch, temp);
        cv::hconcat(temp, e_3ch, row);

        cv::imshow(win_name, row);
        
        char key = (char)cv::waitKey(100);
        if (key == 'q' || key == 'Q') break;

        i = (i + 1) % frames.size();
    }

    cv::destroyAllWindows();
    // Rimosso return 0;
}