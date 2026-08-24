#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

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

// Function that loads all the 72 sequences in memory
// returns a vector of <SequenceData>
// receive as input the main folder path (Sequences' path)
std::vector<SequenceData> load_sequences(const std::string& main_folder_path) {

    // Initialize the vector
    std::vector<SequenceData> all_sequences;
    int seq_counter = 0;

    // Loop through action folders (e.g., boxing, walking...)
    for (const auto& action_folder : fs::directory_iterator(main_folder_path)) {
    // "auto&" figures out the type automatically and reads the original file without heavy copies in memory (hence the &).
    // "const" locks the file in read-only mode, protecting it from accidental changes.
    // "fs::directory_iterator" is an automatic explorer: enters the given folder and at each loop iteration
    //                          automatically delivers the next file or subfolder until none are left. 
    
    if (!fs::is_directory(action_folder.path())) continue;
    // make sure that the found element is a directory (boxing, walking, etc.)
        
        std::string action_name = action_folder.path().filename().string();
        // extracts the folder name, turns it into a string, and saves it in action_name


        // Loop through individual sequences (e.g., person01_boxing_d1...)
        for (const auto& seq_folder : fs::directory_iterator(action_folder.path())) {
        // steps inside the action folder and loops through subfolders of single subjects/scenarios (e.g., person01_boxing_d1)

            if (!fs::is_directory(seq_folder.path())) continue;
            std::string seq_name = seq_folder.path().filename().string();
            // checks again that it is a directory and saves its name in seq_name


            // Extracting person and environment from the folder name
            // Folder name example: person01_boxing_d1 -> persona = person01; env = d1
            std::string persona_name = "";
            std::string env_name = "";
            
            size_t p_pos = seq_name.find("person0");    // looks for "person0" inside the folder name to figure out the person (1, 2, or 3)
            // "size_t" is an integer data type used to represent index positions in a string
            // in this case it returns the position where "person0" was found thanks to .find() (0, because the word is at the very beginning of the folder name)

            if (p_pos != std::string::npos && seq_name.length() >= p_pos + 8) {
            // this if acts as a safety check: 
            // we slice the string ONLY IF "person0" was found and if the string contains enough characters (8)

            // "npos" is the value returned by .find() when the searched word is not found
            // p_pos+8 is the length of "person0n"

                persona_name = seq_name.substr(p_pos, 8); // Grabs e.g. "person01"
            }

            // works similarly to the block above, but rfind (reverse find) starts searching from right to left 
            size_t d_pos = seq_name.rfind("_d");    // looks for "_d" inside the folder name to figure out the environment (1, 2, 3 or 4)
            if (d_pos != std::string::npos && seq_name.length() >= d_pos + 2) {
                env_name = seq_name.substr(d_pos + 1, 2); // Grabs "d1", "d2", etc.
            }


            std::string data_path = seq_folder.path().string() + "/data/";
            // path of the frames folder

            std::vector<cv::String> images_path;
            cv::glob(data_path + "*.png", images_path, false);
            // "glob" looks inside the folder for all files ending in ".png" and automatically collects them into a vector of paths (images_path)

            std::vector<cv::Mat> sequence_frames;   // vector holding the frames
            for (const auto& img_path : images_path) {
                cv::Mat frame = cv::imread(img_path, cv::IMREAD_GRAYSCALE); // reads the file and converts it to grayscale
                if (!frame.empty()) {
                    sequence_frames.push_back(frame);
                    // if the image is read successfully, it gets pushed into the vector
                }
            }

            if (!sequence_frames.empty()) {
                seq_counter++;
                std::string title = action_name + " - " + seq_name;
                
                // save everything into the data structure created at the beginning
                all_sequences.push_back({sequence_frames, action_name, persona_name, env_name, title});
            }
        }
    }

    // Sort all sequences alphabetically based on their complete name
    std::sort(all_sequences.begin(), all_sequences.end(), [](const SequenceData& a, const SequenceData& b) {
        return a.complete_name < b.complete_name;
    });

    std::cout << "[INFO] Successfully loaded " << seq_counter << " sequences (sorted alphabetically)" << std::endl;
    return all_sequences;
}

// Function that allows to receive valid choices
// checks if the input is valid. If not, it skips it
std::string get_valid_choice(const std::string& type){
    std::string input;
    std::vector<std::string> valid_options;
    std::string prompt_msg;

    // Configure valid options based on the requested type
    if (type == "action") {
        valid_options = {"boxing", "handclapping", "handwaving", "walking", "jogging", "running"};
        prompt_msg = "Action (boxing, handclapping, handwaving, walking, jogging, running)? ";
    } 
        else if (type == "person") {
            valid_options = {"person01", "person02", "person03"};
            prompt_msg = "Person (person01, person02, person03)? ";
        } 
        else if (type == "env") {
            valid_options = {"d1", "d2", "d3", "d4"};
            prompt_msg = "Environment / Scenario (d1, d2, d3, d4)? ";
        } 
        else {
            return "";
        }

    std::cout << prompt_msg;
    std::getline(std::cin, input);

    // If the user leaves it blank (presses Enter), skip it
    if (input.empty()) {
        return "";
    }


    // Check if the input matches any of the valid options
    bool is_valid = false;
    for (size_t i = 0; i < valid_options.size(); i++) {
        if (valid_options[i] == input) {
            is_valid = true;
            break; // Found a match, exit the loop
        }
    }

    // If the choice is valid, returns the input; otherwise, warns the user and skips it
    if (is_valid) {
        return input;
    } else {
        std::cout << "[WARNING] Invalid choice! Skipping this choice.\n";
        return "";
    }
}

// Visualization function with choice options for action, person, and environment
// takes as input the vector with all loaded sequences and three strings for what to display
void sequences_visualization(const std::vector<SequenceData>& all_sequences, 
                           std::string action_choice = "", 
                           std::string person_choice = "", 
                           std::string env_choice = "") 
                           {
    int showed = 0;

    for (const auto& seq : all_sequences) {

        // Apply choice filters if specified
        if (!action_choice.empty() && seq.action != action_choice) continue;
        if (!person_choice.empty() && seq.person != person_choice) continue;
        if (!env_choice.empty() && seq.env != env_choice) continue;
        // these ifs check for each sequence if a choice was made:
        // if a decision is present and doesn't match the current sequence data, skip straight to the next one using continue

        showed++;

        std::cout << "Showing: " << seq.complete_name << std::endl;
        
        cv::namedWindow(seq.complete_name, cv::WINDOW_NORMAL); 
        cv::resizeWindow(seq.complete_name, 600, 600); 
        cv::moveWindow(seq.complete_name, 100, 100); // creates the window always at the same screen position
        
        // Display frames of the sequence
        for (const auto& frame : seq.frames) {
            cv::Mat resized_frame;
            cv::resize(frame, resized_frame, cv::Size(640, 480));

            cv::imshow(seq.complete_name, resized_frame);
            char key = (char)cv::waitKey(20); // ~25 FPS
            
            if (key == 'q' || key == 'Q') {
                // added "Q" in case caps lock was active
                std::cout << "\n[INFO] Manual interruption by the user." << std::endl;
                cv::destroyAllWindows();
                return; 
            }
        }
        cv::destroyWindow(seq.complete_name);
    }

    std::cout << "\n[INFO] Visualization completed. " << showed << " sequences displayed." << std::endl;
}


// Function that handles the interactive visualization choice for any dataset
void handle_visualization_menu(const std::vector<SequenceData>& dataset) {
    char answer = 'n';
    std::cout << "Would you like to visualize these sequences? (y/n): ";
    std::cin >> answer;

    if (answer == 'y' || answer == 'Y') {
        int choice = 0;
        std::string action = "";
        std::string person = "";
        std::string env = "";

        std::cout << "\nHow would you like to filter the visualization?\n";
        std::cout << "1) Show ALL sequences\n";
        std::cout << "2) Choose specific ones\n";
        std::cout << "Press 1 or 2: ";
        std::cin >> choice;

        if (choice == 1) {
            std::cout << "\n[INFO] Displaying all sequences...\n";
            sequences_visualization(dataset, action, person, env);
        } 
        else if (choice == 2) {
            std::cout << "\nEnter your choice:\n";
            std::cin.ignore();  

            action = get_valid_choice("action");
            person = get_valid_choice("person");
            env = get_valid_choice("env");

            std::cout << "\n[INFO] Starting filtered visualization...\n";
            sequences_visualization(dataset, action, person, env);
        }
    } else {
        std::cout << "\n[INFO] Skipping sequences visualization.\n";
    }
}

// Phase 1: Loads the dataset and optionally lets the user view the raw sequences
std::vector<SequenceData> run_phase1() {
    std::string sequences_path = "../Sequences";

    std::cout << "[INFO] Starting sequence loading..." << std::endl;
    std::vector<SequenceData> dataset = load_sequences(sequences_path);

    if (dataset.empty()) {
        std::cerr << "[ERROR] No sequence found! Check the path of the 'Sequences' folder." << std::endl;
        return {}; 
    }

    std::cout << "[INFO] Loading completed." << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    handle_visualization_menu(dataset);

    return dataset;
}

