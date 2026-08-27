#include <iostream>
#include "tecniche_segmentazione.h"

int main() {
    int choice = -1;

    // Quick and dirty menu to test all the pipelines without recompiling everything every time.
    // Make sure the "data" folder is in the same directory as the executable, 
    // otherwise the functions will just return and do nothing!
    
    while (choice != 0) {
        std::cout << "\n=========================================================\n";
        std::cout << "   COMPUTER VISION PROJECT - SEGMENTATION TESTS MENU     \n";
        std::cout << "=========================================================\n";
        std::cout << "1. Comparison (Otsu vs Watershed vs MeanShift)\n";
        std::cout << "2. Otsu-guided Watershed (with edge dilation trackbar)\n";
        std::cout << "3. Edge Detection (Sobel + Otsu)\n";
        std::cout << "4. Contrast stretching test (Stretching + Sobel + Otsu)\n";
        std::cout << "5. Otsu Grid 2x2 with Morphological Cleaning\n";
        std::cout << "0. Exit\n";
        std::cout << "---------------------------------------------------------\n";
        std::cout << "Select a test to run (0-5): ";
        
        std::cin >> choice;

        // Clear input buffer just in case we type a letter by mistake
        if(std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            std::cout << "[ERROR] Invalid input. Please enter a number.\n";
            continue;
        }

        std::cout << "\n";

        // Switch to call the right function based on user input
        switch (choice) {
            case 1:
                std::cout << "--> Starting Final Comparison...\n";
                // Best results so far!
                comparison_otsu_watershed_meanshift();
                break;
            case 2:
                std::cout << "--> Starting Otsu + Watershed...\n";
                test_otsu_then_watershed();
                break;
            case 3:
                std::cout << "--> Starting Sobel + Otsu...\n";
                test_sobel_then_otsu();
                break;
            case 4:
                std::cout << "--> Starting Stretching + Sobel + Otsu...\n";
                // Still tweaking the LUT trackbars for this one
                test_stretching_then_sobel_then_otsu();
                break;
            case 5:
                std::cout << "--> Starting Otsu Grid 2x2...\n";
                test_otsu_4regions();
                break;
            case 0:
                std::cout << "Exiting. Bye!\n";
                break;
            default:
                std::cout << "[ERROR] Wrong number, try again.\n";
                break;
        }
    }

    return 0;
}