//This algorithm is based on a simply idea. In "dynamic" sequences (e.g. walking, jogging, running), the actor occupy a given field for only a few moments;
//then, in the remaing seconds, the background behind him is "free" from him. Therefore, we can compute the "median" background frame, as the original background,
// and trying to subtract it from each frame in order to have the isolated actor (foreground). The median frame is composed by the median pixel values
//computed at all frames coordinates.
//This algorithm fails when actor is stands still in a frame area and moves only arms or hands (e.g. handwaving, handclapping, boxing),
//because we are not able to compute the background behind him. In that case, the algorithm subtract also the actor as it is part of the background.
//Another serious limit of this approach is about shadows: often they are considered as "foreground" elements. Maybe it can be controlled by pre-processing. 

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp> 
#include <vector>
#include <iostream>
#include <string>
#include <algorithm> 
#include <iomanip>   //It was useful only to load frames. To be deleted when we'll intgrate this code with yours about frames loading.

using namespace std;

int main() {
    
    string output_folder = "results"; //the directory containg the output (processed) frames
    if (!cv::utils::fs::exists(output_folder)) { //if it does not exist, then create it
        cv::utils::fs::createDirectory(output_folder);
    }
    
    // 1) FRAMES LOADING - TO BE DELETED
    
    vector<cv::Mat> frames;
    
    for (int i = 1; i <= 40; i++) {
        stringstream ss;
        ss << "input/frame_" << setfill('0') << setw(2) << i << ".png"; //<iomanip> is used just here; this program reads directly from a directory called 'input'
        string file_name = ss.str();
        
        cv::Mat img = cv::imread(file_name, 0); 
        
        if (img.empty()) {
            cout << "Error: file not found " << file_name << endl;
            return -1;
        }
        frames.push_back(img);
    }

    // 2) GENERATING MEDIAN BACKGROUND

    int height = frames[0].rows; //frame height
    int width = frames[0].cols; //frame width
    
    cv::Mat background_img = cv::Mat::zeros(height, width, CV_8UC1); //initializing a void (black) image having same dimension as input frames

    for (int r = 0; r < height; r++) { //scanning the frame(s) rows and columns
        for (int c = 0; c < width; c++) {
            vector<int> pixel_values; //pixel_values vector definition
            for (int f = 0; f < 40; f++) {                 //for each frame
                int greylevel = frames[f].at<uchar>(r, c); //storing pixel value at coordinates (r,c) in the pixel_values vector
                pixel_values.push_back(greylevel);
            }
            std::sort(pixel_values.begin(), pixel_values.end()); //sorting pixel_values vector
            int median = pixel_values[20]; //median value (among pixel values stored)
            background_img.at<uchar>(r, c) = median; //background image pixels values assignment
        }
    }
    
    // 3) GENERATING BINARY MASKS
    
    cv::Mat struct_el = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)); //structuring element definition

    cout << "Starting masks generation" << endl;

    for (int i = 1; i <= 40; i++) { //cycle on all frames
        
        cv::Mat current_frame = frames[i - 1];

        cv::Mat diff_map;
        cv::absdiff(current_frame, background_img, diff_map); //performing difference between frames and the computed background image

        cv::GaussianBlur(diff_map, diff_map, cv::Size(3, 3), 0); //performing a Gaussian blur (optional or editable)

        cv::Mat binary_mask;
        cv::threshold(diff_map, binary_mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU); //Otsu's optimal threshold computation on diff_map

        // Closing morphological operation in order to close holes
        cv::Mat final_mask;
        cv::morphologyEx(binary_mask, final_mask, cv::MORPH_CLOSE, struct_el); //closing on binary_mask

        // Saving output frames in a separate directory. It was useful to me for output examination, to be delated and replaced by frames sequence visualization
        stringstream ss_out;
        ss_out << output_folder << "/mask_frame_" << setfill('0') << setw(2) << i << ".png"; //each output frame represents the binary mask of the corresponding input frame (same number in the name)
        cv::imwrite(ss_out.str(), final_mask); //storing final binary masks
    }

    cout << "Output frames have been stored in '" << output_folder << "/'." << endl;
    return 0;
}