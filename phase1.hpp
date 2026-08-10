#ifndef PHASE1_HPP
#define PHASE1_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Data Structure containing sequence's informations
struct SequenceData {
    std::vector<cv::Mat> frames;
    std::string action;
    std::string persona;
    std::string env;
    std::string nome_completo;
};

// Function that loads all the sequences 
std::vector<SequenceData> load_sequences(const std::string& main_folder_path);


// Function that allows to receive valid choices
std::string get_valid_choice(const std::string& type);


// Function that display a/some chosen sequences
void sequences_visualization(const std::vector<SequenceData>& tutte_le_sequenze, 
                             std::string action_filter = "", 
                             std::string persona_filter = "", 
                             std::string env_filter = "");




#endif // PHASE1_HPP