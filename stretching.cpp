#include <iostream>
#include <opencv2/opencv.hpp>

// Global variables for trackbars and real-time processing
cv::Mat frame_global, bilateral_filtered_global;
double scale_factor_global = 2.0; // Scaled to 2.0 to fit 6 panels comfortably on screen

// Trackbar parameters
int r1_val = 88;
int s1_val = 15;
int r2_val = 163;
int s2_val = 255;

// Function to print a short explanation of the sliders' role
void printSliderGuide() {
    std::cout << "\n--------------------------------------------------\n";
    std::cout << "[SLIDER GUIDE & CURRENT VALUES]:\n";
    std::cout << " - r1 (" << r1_val << "): Input threshold for dark tones.\n";
    std::cout << " - s1 (" << s1_val << "): Output intensity mapped to r1.\n";
    std::cout << " - r2 (" << r2_val << "): Input threshold for mid/bright tones.\n";
    std::cout << " - s2 (" << s2_val << "): Output intensity mapped to r2.\n";
    std::cout << "--------------------------------------------------\n";
}

// Function to update the 6-panel pipeline in real time
void updatePipeline(int, void*) {
    if (r1_val > r2_val) r1_val = r2_val;
    if (r1_val < 0) r1_val = 0;
    if (r2_val > 255) r2_val = 255;
    if (s1_val < 0) s1_val = 0;
    if (s1_val > 255) s1_val = 255;
    if (s2_val < 0) s2_val = 0;
    if (s2_val > 255) s2_val = 255;

    printSliderGuide();

    // 1. Build the Lookup Table (LUT) for piecewise linear stretching
    cv::Mat lut(1, 256, CV_8U);
    uchar* p_lut = lut.data;

    for (int r = 0; r < 256; r++) {
        double s = 0.0;
        if (r < r1_val) {
            s = (r1_val > 0) ? ((double)s1_val / r1_val) * r : 0;
        } else if (r >= r1_val && r <= r2_val) {
            s = (r2_val != r1_val) ? s1_val + ((double)(s2_val - s1_val) / (r2_val - r1_val)) * (r - r1_val) : s1_val;
        } else {
            s = (255 != r2_val) ? s2_val + ((double)(255 - s2_val) / (255 - r2_val)) * (r - r2_val) : s2_val;
        }
        p_lut[r] = (uchar)(cv::saturate_cast<uchar>(s));
    }

    // --- RAMO A: APPLICATO SULL'IMMAGINE ORIGINALE ---
    cv::Mat stretched_orig;
    cv::LUT(frame_global, lut, stretched_orig);

    cv::Mat grad_x_orig, grad_y_orig, sobel_orig;
    cv::Sobel(stretched_orig, grad_x_orig, CV_16S, 1, 0, 3);
    cv::Sobel(stretched_orig, grad_y_orig, CV_16S, 0, 1, 3);
    cv::Mat abs_x_orig, abs_y_orig;
    cv::convertScaleAbs(grad_x_orig, abs_x_orig);
    cv::convertScaleAbs(grad_y_orig, abs_y_orig);
    cv::addWeighted(abs_x_orig, 0.5, abs_y_orig, 0.5, 0, sobel_orig);

    // --- RAMO B: APPLICATO SUL BILATERAL ---
    cv::Mat stretched_bilat;
    cv::LUT(bilateral_filtered_global, lut, stretched_bilat);

    cv::Mat grad_x_bilat, grad_y_bilat, sobel_bilat;
    cv::Sobel(stretched_bilat, grad_x_bilat, CV_16S, 1, 0, 3);
    cv::Sobel(stretched_bilat, grad_y_bilat, CV_16S, 0, 1, 3);
    cv::Mat abs_x_bilat, abs_y_bilat;
    cv::convertScaleAbs(grad_x_bilat, abs_x_bilat);
    cv::convertScaleAbs(grad_y_bilat, abs_y_bilat);
    cv::addWeighted(abs_x_bilat, 0.5, abs_y_bilat, 0.5, 0, sobel_bilat);

    // --- PREPARAZIONE DEI 6 PANNELLI IN BGR ---
    cv::Mat b_orig, b_stretched_orig, b_sobel_orig;
    cv::Mat b_bilat, b_stretched_bilat, b_sobel_bilat;

    cv::cvtColor(frame_global, b_orig, cv::COLOR_GRAY2BGR);
    cv::cvtColor(stretched_orig, b_stretched_orig, cv::COLOR_GRAY2BGR);
    cv::cvtColor(sobel_orig, b_sobel_orig, cv::COLOR_GRAY2BGR);

    cv::cvtColor(bilateral_filtered_global, b_bilat, cv::COLOR_GRAY2BGR);
    cv::cvtColor(stretched_bilat, b_stretched_bilat, cv::COLOR_GRAY2BGR);
    cv::cvtColor(sobel_bilat, b_sobel_bilat, cv::COLOR_GRAY2BGR);

    // Organizzazione in griglia 2x3 (Riga 1: Originale | Orig Stretched | Sobel Orig)
    //              (Riga 2: Bilateral | Bilat Stretched | Sobel Bilat)
    cv::Mat row1_left, row1_right, row1_final;
    cv::hconcat(b_orig, b_stretched_orig, row1_left);
    cv::hconcat(row1_left, b_sobel_orig, row1_final);

    cv::Mat row2_left, row2_right, row2_final;
    cv::hconcat(b_bilat, b_stretched_bilat, row2_left);
    cv::hconcat(row2_left, b_sobel_bilat, row2_final);

    cv::Mat grid_display;
    cv::vconcat(row1_final, row2_final, grid_display);

    cv::Mat display_resized;
    cv::resize(grid_display, display_resized, cv::Size(), scale_factor_global, scale_factor_global, cv::INTER_LINEAR);

    cv::imshow("6-Panel Pipeline: Original vs Bilateral Comparison", display_resized);
}

int main() {
    std::string image_path = "frame_20.png";
    frame_global = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    if (frame_global.empty()) {
        std::cerr << "[ERROR] Could not load image: " << image_path << std::endl;
        return -1;
    }

    // Pre-calculate Bilateral filter
    cv::bilateralFilter(frame_global, bilateral_filtered_global, 10, 20.0, 20.0);

    std::string window_name = "6-Panel Pipeline: Original vs Bilateral Comparison";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    cv::createTrackbar("r1", window_name, &r1_val, 255, updatePipeline);
    cv::createTrackbar("s1", window_name, &s1_val, 255, updatePipeline);
    cv::createTrackbar("r2", window_name, &r2_val, 255, updatePipeline);
    cv::createTrackbar("s2", window_name, &s2_val, 255, updatePipeline);

    updatePipeline(0, 0);

    std::cout << "[INFO] 6-panel grid running. Check how stretching impacts raw vs bilateral images. Press 'q' to exit." << std::endl;

    while (true) {
        char key = (char)cv::waitKey(30);
        if (key == 'q' || key == 'Q') {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}