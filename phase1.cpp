#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// Struttura per contenere i dati di una singola sequenza e i suoi tag
struct SequenceData {
    std::vector<cv::Mat> frames;
    std::string action;
    std::string person;
    std::string env;
    std::string complete_name;
};

// Funzione per caricare tutte le 72 sequenze in memoria
// restituisce un vettore di SequenceData (le sequenze con i vari tag (azione, persona, environment))
// riceve in ingresso il percorso della cartella principale (Sequences)
std::vector<SequenceData> load_sequences(const std::string& main_folder_path) {

    // inizializzazione del vettore che conterrà le 72 sequenze
    std::vector<SequenceData> all_sequences;
    int seq_counter = 0;

    // Ciclo sulle cartelle delle azioni (es. boxing, walking...)
    for (const auto& action_folder : fs::directory_iterator(main_folder_path)) {
    // "auto&" capisce il tipo da solo e legge il file originale senza fare copie pesanti in memoria (per questo il &).
    // "const" blocca il file in sola lettura, proteggendolo da modifiche accidentali.
    // "fs::directory_iterator" è un esploratore automatico: entra nella cartella indicata e ad ogni ciclo del for
    //                          consegna in automatico il file o la sottocartella successiva finchè ne esistono. 
    
    if (!fs::is_directory(action_folder.path())) continue;
    // mi assicuro che l'elemento trovato sia una cartella (boxing, walking, ecc..)
        
        std::string action_name = action_folder.path().filename().string();
        // estrae il nome della cartella trasformandolo in stringa e lo salva in action_name


        // Ciclo sulle singole sequenze (es. person01_boxing_d1...)
        for (const auto& seq_folder : fs::directory_iterator(action_folder.path())) {
        // entra dentro la cartella dell'azione e scorre le sottocartelle dei singoli soggetti/scenari (es. person01_boxing_d1)

            if (!fs::is_directory(seq_folder.path())) continue;
            std::string seq_name = seq_folder.path().filename().string();
            // anche qui controlla che sia una cartella e ne salva il nome in seq_name


            // Estrazione della persona e l'environment dal nome della cartella
            // Esempio nome cartella: person01_boxing_d1 -> persona = person01; env = d1
            std::string persona_name = "";
            std::string env_name = "";
            
            size_t p_pos = seq_name.find("person0");    // cerca "person0" dentro il nome della cartella per capire la persona (1, 2, o 3)
            // "size_t" è un tipo di dato intero per usato rappresentare l'indice di posizione in una stringa
            // in questo caso restituisce la posizione in cui è stato trovato "person0" grazie a .find() (0, perchè la parola è all'inizio del nome della cartella)

            if (p_pos != std::string::npos && seq_name.length() >= p_pos + 8) {
            // questo if serve per fare un controllo di sicurezza: 
            // tagliamo la stringa SOLO SE è stato trovato "person0" e se la stringa contiene abbastanza caratteri (8)

            // "npos" è il valore restituito da .find() quando non viene trovata la parola che cerca
            // p_pos+8 è la lunghezza di "person0n"

                persona_name = seq_name.substr(p_pos, 8); // Prende ad esempio "person01"
            }

            // funziona in modo simile al pezzo sopra ma rfind (reverse find) inizia la ricerca da destra verso sinistra 
            size_t d_pos = seq_name.rfind("_d");    // cerca "_d" dentro il nome della cartella per capire l'environment (1, 2, 3 o 4)
            if (d_pos != std::string::npos && seq_name.length() >= d_pos + 2) {
                env_name = seq_name.substr(d_pos + 1, 2); // Prende "d1", "d2", ecc.
            }


            std::string data_path = seq_folder.path().string() + "/data/";
            // percorso della cartella dei frames

            std::vector<cv::String> images_path;
            cv::glob(data_path + "*.png", images_path, false);
            // "glob" cerca nella cartella tutti i file che finiscono con ".png" e li raccoglie in automatico dentro un vettore di percorsi (images_path)

            std::vector<cv::Mat> sequence_frames;   // vettore che contiene i frame
            for (const auto& img_path : images_path) {
                cv::Mat frame = cv::imread(img_path, cv::IMREAD_GRAYSCALE); // legge il file e lo converte in grayscale
                if (!frame.empty()) {
                    sequence_frames.push_back(frame);
                    // se l'immagine è letta correttamente, viene inserita nel vettore
                }
            }

            if (!sequence_frames.empty()) {
                seq_counter++;
                std::string title = action_name + " - " + seq_name;
                
                // salviamo tutto nella struttura dati creata all'inizio
                all_sequences.push_back({sequence_frames, action_name, persona_name, env_name, title});
            }
        }
    }

    std::cout << "[INFO] Successfully loaded " << seq_counter << " sequences" << std::endl;
    return all_sequences;
}

// Funzione di visualizzazione con opzione di scelta per azione, persona ed environment
// prende in input il vettore con tutte le sequenze caricate e tre stringhe per la scelta di cosa visualizzare
void sequences_visualization(const std::vector<SequenceData>& all_sequences, 
                             std::string action_choice = "", 
                             std::string persona_choice = "", 
                             std::string env_choice = "") 
                             {
    int showed = 0;

    for (const auto& seq : all_sequences) {
        
        // Applicazione delle scelte (se specificate)
        if (!action_choice.empty() && seq.action != action_choice) continue;
        if (!persona_choice.empty() && seq.person != persona_choice) continue;
        if (!env_choice.empty() && seq.env != env_choice) continue;
        // questi if controllano per ogni sequenza se è stata fatta una scelta:
        // se è presente una decisione e non corrisponde ai dati della sequenza corrente, salta subito alla successiva grazie a continue

        showed++;

        std::cout << "Showing: " << seq.complete_name << std::endl;
        

        cv::namedWindow(seq.complete_name, cv::WINDOW_NORMAL); // crea una finestra ridimensionabile
        cv::resizeWindow(seq.complete_name, 600, 600); // imposta una dimensione ingrandita
        cv::moveWindow(seq.complete_name, 100, 100); // crea la finestra sempre nello stesso punto

        // Mostra i frame della sequenza a video
        for (const auto& frame : seq.frames) {
            cv::Mat resized_frame;
            cv::resize(frame, resized_frame, cv::Size(640, 480));

            cv::imshow(seq.complete_name, resized_frame);
            char key = (char)cv::waitKey(40); // 25 FPS
            
            if (key == 'q' || key == 'Q') {
                // aggiunto "Q" nel caso in cui si avesse il blocco maiuscole attivo
                std::cout << "\n[INFO] Manual interruption by the user ('q')." << std::endl;
                cv::destroyAllWindows();
                return; 
            }
        }
        cv::destroyWindow(seq.complete_name);
    }

    std::cout << "[INFO] Visualization completed. Showed " << showed << " chosen sequences " << std::endl;
}