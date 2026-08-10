#include <iostream>
#include <vector>
#include "phase1.hpp" // includiamo l'header della fase 1

int main() {
    std::string sequences_path = "../Sequences";

    std::cout << "[INFO] Starting sequence loading..." << std::endl;
    
    // 1. Carichiamo tutto in memoria all'avvio
    std::vector<SequenceData> dataset = load_sequences(sequences_path);

    // Controlliamo se è stato caricato qualcosa
    if (dataset.empty()) {
        std::cerr << "[ERROR] No sequence found! Check the path of the 'Sequences' folder." << std::endl;
        return -1;
    }

    std::cout << "[INFO] Loading completed." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // 2. Interfaccia interattiva con l'utente 
    int choice = 0;
    // inizializzazione stringhe
    std::string action = "";
    std::string person = "";
    std::string env = "";

    std::cout << "How would you like to proceed with the visualization?\n";
    std::cout << "1) Show ALL sequences\n";
    std::cout << "2) Choose some of them\n";
    std::cout << "Press 1 or 2: ";
    std::cin >> choice;

    if (choice == 2) {
        std::cout << "\nEnter the choices (leave blank or press enter to skip, or type the value):\n";
        
        std::cin.ignore();  // pulisce il buffer del terminale prima di leggere le stringhe

        std::cout << "Action (boxing, walking, running...)? ";
        std::getline(std::cin, action);

        std::cout << "Person (es. person01, person02...)? ";
        std::getline(std::cin, person);

        std::cout << "Environment / Scenario (es. d1, d2, d3, d4)? ";
        std::getline(std::cin, env);
    } else {
        std::cout << "\n[INFO] No choices applied. All sequences will be displayed.\n";
    }

    std::cout << "\n[INFO] Starting visualization..." << std::endl;
    std::cout << "[INFO] Press 'q' or 'Q' during visualization to stop.\n" << std::endl;

    // 3. Chiamata alla funzione passando le variabili inserite tramite cin
    sequences_visualization(dataset, action, person, env);

    std::cout << "\n=== PROGRAM TERMINATED SUCCESSFULLY ===" << std::endl;

    return 0;
}