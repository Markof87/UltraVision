#ifndef PHASE1_HPP
#define PHASE1_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Struttura dati per contenere i frame e i relativi tag della sequenza
struct SequenceData {
    std::vector<cv::Mat> frames;
    std::string action;
    std::string persona;
    std::string env;
    std::string nome_completo;
};

// Dichiarazione della funzione di caricamento delle sequenze
std::vector<SequenceData> load_sequences(const std::string& main_folder_path);


// Dichiarazione della funzione di visualizzazione con possibile scelta
void sequences_visualization(const std::vector<SequenceData>& tutte_le_sequenze, 
                             std::string action_filter = "", 
                             std::string persona_filter = "", 
                             std::string env_filter = "");


#endif // PHASE1_HPP