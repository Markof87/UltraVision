#include <iostream>
#include <vector>
#include <string>
#include <iomanip> // Per std::setw e std::setfill
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "[INFO] Loading and filtering the 40 frames..." << std::endl;

    std::vector<cv::Mat> filtered_sequence;
    double scale_factor = 3.0; // Ingrandimento 3x per vederli bene

    // 1. Caricamento e filtraggio dei 40 frame
    for (int i = 1; i <= 40; i++) {
        // Genera il nome con lo zero iniziale per i numeri da 1 a 9 (es. frame_01.png, ..., frame_09.png, frame_10.png)
        std::ostringstream filename_stream;
        filename_stream << "frame_" << std::setw(2) << std::setfill('0') << i << ".png";
        std::string filename = filename_stream.str();
        
        cv::Mat frame = cv::imread(filename, cv::IMREAD_GRAYSCALE);
        if (frame.empty()) {
            std::cerr << "[WARNING] Could not load: " << filename << ". Stopping sequence load." << std::endl;
            break;
        }

        // Applica il Bilateral Filter con d=10, sigmaColor=20, sigmaSpace=20
        cv::Mat filtered;
        cv::bilateralFilter(frame, filtered, 10, 20.0, 20.0);

        // Ingrandisce il frame filtrato per una visualizzazione ottimale
        cv::Mat frame_resized;
        cv::resize(filtered, frame_resized, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);

        filtered_sequence.push_back(frame_resized);
    }

    if (filtered_sequence.empty()) {
        std::cerr << "[ERROR] No frames were loaded. Check file names and path." << std::endl;
        return -1;
    }

    std::cout << "[INFO] Loading complete! Playing sequence. Press 'q' to exit." << std::endl;

    // 2. Creazione della finestra di visualizzazione
    cv::namedWindow("Filtered Sequence Player", cv::WINDOW_AUTOSIZE);

    // 3. Riproduzione in loop dei frame (simula il video a circa 25 FPS -> 40ms di attesa per frame)
    int current_frame = 0;
    while (true) {
        // Mostra il frame corrente
        cv::imshow("Filtered Sequence Player", filtered_sequence[current_frame]);

        // Attende 40 millisecondi prima di passare al frame successivo
        char key = (char)cv::waitKey(40);
        
        // Se l'utente preme 'q' o 'Q', esce dal ciclo
        if (key == 'q' || key == 'Q') {
            break;
        }

        // Passa al frame successivo, tornando all'inizio quando arriva a 40 (loop continuo)
        current_frame = (current_frame + 1) % filtered_sequence.size();
    }

    cv::destroyAllWindows();
    std::cout << "[INFO] Playback finished." << std::endl;

    return 0;
}