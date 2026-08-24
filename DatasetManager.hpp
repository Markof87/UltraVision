#ifndef DATASETMANAGER_HPP
#define DATASETMANAGER_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Data Structure containing sequence's informations
struct SequenceData {
    std::vector<cv::Mat> frames;
    std::string action;
    std::string person;
    std::string env;
    std::string complete_name;
};

//-----------------------------------------------------------------------------------------------------------------
// PHASE1: SEQUENCES LOADING AND VISUALIZATION
//-----------------------------------------------------------------------------------------------------------------

// Function that loads all the sequences 
std::vector<SequenceData> load_sequences(const std::string& main_folder_path);


// Function that allows to receive valid choices
std::string get_valid_choice(const std::string& type);


// Function that displays some chosen sequences
void sequences_visualization(const std::vector<SequenceData>& all_sequences, 
                             std::string action_filter = "", 
                             std::string person_filter = "", 
                             std::string env_filter = "");

//Function that asks the user if he wants to visualize the current sequences
void handle_visualization_menu(const std::vector<SequenceData>& dataset);

// Function that runs the entire phase1
std::vector<SequenceData> run_phase1();


#endif 