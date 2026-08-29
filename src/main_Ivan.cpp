#include <iostream>
#include <vector>
#include <filesystem> // La libreria utile per le cartelle
#include <opencv2/opencv.hpp>

#include "utils.hpp" // Il file ausiliario

namespace fs = std::filesystem;

int main() {
    // Cartella principale (il ../ serve per uscire da build)
    std::string main_folder = "../Sequences";

    // Variabile per contare quante sequenze abbiamo letto in totale (dovranno essere 72)
    int seq_counter = 0;

    // Primo ciclo: entriamo nelle 6 cartelle delle azioni (boxing, walking, ecc.)
    for (const auto& action_folder : fs::directory_iterator(main_folder)) {
        // "auto&" capisce il tipo da solo e legge il file originale senza fare copie pesanti in memoria (per questo il &).
        // "const" blocca il file in sola lettura, proteggendolo da modifiche accidentali.
        // "fs::directory_iterator" è un esploratore automatico: entra nella cartella indicata e ad ogni ciclo del for,
        //                          consegna in automatico il file o la sottocartella successiva finchè ne esistono. 


        // Secondo ciclo: entriamo nelle 12 sequenze (person01_d1, person02_d2, ecc.)
        for (const auto& seq_folder : fs::directory_iterator(action_folder.path())) {
            // Qui usiamo .path() per estrarre da "action_folder" SOLO l'indirizzo di testo puro, ad esempio
            // "../Sequences/boxing" che viene consegnato al nuovo esploratore, in modo da entrare all'interno della cartella "boxing".


            // Costruiamo il percorso esatto della cartella "data" per la sequenza corrente
            std::string data_path = seq_folder.path().string() + "/data/";
            // Ad esempio, si otterrà la stringa "../Sequences/boxing/data/"
            
            
            // Creiamo i vettori DENTRO il ciclo. 
            // In questo modo ripartono da zero ad ogni nuova sequenza
            std::vector<cv::String> images_path;
            cv::glob(data_path + "*.png", images_path, false);
            // "cv::glob" cerca nella cartella tutti i file che finiscono con ".png" e li raccoglie in automatico dentro un vettore di percorsi.     

            std::vector<cv::Mat> sequence_frames;
            // Usiamo un vector in modo che il "dimensionamento" avvenga in automatico (a differenza degli array)

            // Carichiamo i famosi 40 fotogrammi
            for (size_t i = 0; i < images_path.size(); i++) {
                cv::Mat frame = cv::imread(images_path[i], cv::IMREAD_GRAYSCALE);
                if (!frame.empty()) {
                    sequence_frames.push_back(frame);
                    // "push_back" aggiunge il nuovo frame appena letto in fondo al vettore, allungandolo di uno spazio.
                }
            }

            // --- A QUESTO PUNTO LE IMMAGINI SONO ISOLATE ---
            // sequence_frames contiene ESATTAMENTE i 40 frame di questa specifica cartella.
            
            // Incrementiamo il contatore
            seq_counter++;

            // Creiamo una finestra con titolo dinamico (con il nome dell'azione e della persona)
            std::string nome_finestra = action_folder.path().filename().string() + " - " + seq_folder.path().filename().string();
            bool continua = show_sequence(sequence_frames, nome_finestra);  // Funzione ausiliaria per stampare il "video"
            if (!continua) {
                std::cout << "\nInterruzione manuale dell'utente" << std::endl;
                std::cout << "Passaggio alla prossima azione\n" << std::endl;
                break; // Esce dal ciclo delle sequenze
            }

            // Stampiamo a schermo cosa sta facendo il programma in tempo reale
            std::cout << "Analyzed sequence N. " << seq_counter
                      << " | Action: " << action_folder.path().filename().string() 
                      << " | File: " << seq_folder.path().filename().string() 
                      << " | Frame: " << sequence_frames.size() << std::endl;
            
            // *** QUI IN FUTURO INSERIREMO LE FUNZIONI DELLA FASE 2 E 3 ***
            // Esempio: calcolaBoundingBox(sequence_frames);
            // Esempio: estraiFeatures(sequence_frames);

        } // <-- Fine del ciclo della singola sequenza. 
          // Qui la variabile 'sequence_frames' viene distrutta. Impossibile fare confusione!
    }

    std::cout << "\n=== FASE 1 COMPLETATA ===" << std::endl;
    std::cout << "Totale sequenze elaborate: " << seq_counter << " (In totale sono 72)" << std::endl;

    return 0;
}