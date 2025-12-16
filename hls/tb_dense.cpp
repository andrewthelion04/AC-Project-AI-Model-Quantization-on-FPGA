#include <iostream>
#include "test_image.h" // Trebuie sa existe in acelasi folder

// Declaram functia din dense_int8.cpp
void dense_int8(const float input_image[784], float output_scores[10]);

int main() {
    float scores[10];

    std::cout << "Starting HLS Simulation..." << std::endl;
    
    // Rulam functia hardware
    dense_int8(test_image, scores);

    // Afisam rezultatele
    std::cout << "Output Scores:" << std::endl;
    float max_val = -99999.0f;
    int pred = -1;

    for(int i=0; i<10; i++) {
        std::cout << "Digit " << i << ": " << scores[i] << std::endl;
        if (scores[i] > max_val) {
            max_val = scores[i];
            pred = i;
        }
    }

    std::cout << "PREDICTION: " << pred << std::endl;
    std::cout << "EXPECTED:   " << expected_label << std::endl;

    if (pred == expected_label) {
        std::cout << "TEST PASS!" << std::endl;
        return 0;
    } else {
        std::cout << "TEST FAIL!" << std::endl;
        return 1;
    }
}