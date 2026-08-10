#include <iostream>
#include <vector>
#include "phase1.hpp" // Include the Phase 1's header file

int main() {
    std::string sequences_path = "../Sequences";

    std::cout << "[INFO] Starting sequence loading..." << std::endl;
    
    // 1. Load everything into memory at the beginning
    std::vector<SequenceData> dataset = load_sequences(sequences_path);

    // Check if anything was loaded
    if (dataset.empty()) {
        std::cerr << "[ERROR] No sequence found! Check the path of the 'Sequences' folder." << std::endl;
        return -1;
    }

    std::cout << "[INFO] Loading completed." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    

    // 2. Interactive User Interface
    int choice = 0;

    // Initialize strings for choices
    std::string action = "";
    std::string person = "";
    std::string env = "";

    std::cout << "How would you like to proceed with the visualization?\n";
    std::cout << "1) Show ALL sequences\n";
    std::cout << "2) Choose some of them\n";
    std::cout << "Press 1 or 2: ";
    std::cin >> choice;

    if (choice == 2) {
        std::cout << "\nEnter your choice (leave blank or press enter to skip, or type the value):\n";
        std::cin.ignore();  // clears the terminal buffer before reading strings

        // Get valid choices one by one using the auxiliary function
        action = get_valid_choice("action");
        person = get_valid_choice("person");
        env = get_valid_choice("env");
    } else {
        std::cout << "\n[INFO] No choices applied. All sequences will be displayed.\n";
    }

    std::cout << "\n[INFO] Starting visualization..." << std::endl;
    std::cout << "[INFO] Press 'q' or 'Q' during visualization to stop.\n" << std::endl;

    // 3. Call the funztion passing the variables entered via cin
    sequences_visualization(dataset, action, person, env);

    std::cout << "\n=== PROGRAM TERMINATED SUCCESSFULLY ===" << std::endl;

    return 0;
}