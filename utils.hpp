#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Funzione per mostrare a video una sequenza di frame
inline bool show_sequence(const std::vector<cv::Mat>& frames, const std::string& window_title) {
// inline serve per fare una sorta di "copia e incolla" senza tornare ogni volta dentro utils (ottimizzando le prestazioni)
// "frames" è il pacchetto che contiene tutti i 40 fotogrammi della sequenza che la funzione deve mostrare a schermo.
// "window_title" è il testo personalizzato (es. "boxing - person1") che apparirà scritto sulla barra in alto della finestra video.

    // Se il vettore è vuoto, evitiamo errori
    if (frames.empty()) return true;

    std::cout << "[INFO] Premi 'q' sulla finestra video per interrompere" << std::endl;

    for (size_t i = 0; i < frames.size(); i++) {
        cv::Mat frame_ingrandito;
        
        // Ingrandiamo l'immagine per vederla comoda (da 160x120 a 640x480)
        cv::resize(frames[i], frame_ingrandito, cv::Size(640, 480));
        
        // Mostriamo il frame con il titolo personalizzato (es. nome dell'azione/persona)
        cv::imshow(window_title, frame_ingrandito);
        
        if (i == 0) {
            cv::moveWindow(window_title, 500, 200); // Cambia 500 e 200 in base a dove la vuoi sullo schermo
        }
        

        if (cv::waitKey(60) == 'q') {
            cv::destroyWindow(window_title); 
            return false; 
            // premendo 'q', restituiamo 'false' per segnalare che vogliamo interrompere il programma
            // considerando il ciclo all'interno del main, si passerà all'azione successiva
        }
    }

    // Appena il video dei 40 frame finisce, distruggiamo la finestra di questa specifica sequenza
    cv::destroyWindow(window_title);
    return true;
}

#endif