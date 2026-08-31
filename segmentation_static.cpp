//This algorithm is based on thresolded gradient of the smoothed image + Hough transform + morphological operations.
//Thresholded gradient gives the edges (silhouette) of the actor in a binary mask (thresholded) but the silhouette is not a "close" line, it has several holes,
//then I tried close the shape with the Hough Transform, a technique capable of linking edges. By the way, Hough produced thick edges,
//then I performed a series of morphological operations in order to obtain a more closed and thinner silhouette.
//Morphological operators are useful also to remove the "floor line" and other noisy elements in the background, enhanced by derivative operator.
//It works well on "static" sequences (e.g. boxing, handwaving, handclapping) and it's still acceptable on "dynamic" ones (e.g. walking, jogging, running).
//The main fault of this approach is given by the shape of the actor silhouette: it is "void" (black) in some little areas in the upper body.
//It is not so important, for the segmentation task, to have a perfectly solid silhouette (filled, completely white) but it's fair to highlight it.
//Moreover, this approach is based only on a combination of "simply" techniques already seen during the course.

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp> 
#include <vector>
#include <iostream>
#include <string>
#include <algorithm> 
#include <iomanip>  //It was useful only to load frames. To be deleted when we'll intgrate this code with yours about frames loading.
 

using namespace std;

void createDirectory(const string& path) {
    if (!cv::utils::fs::exists(path)) {
        cv::utils::fs::createDirectory(path);
    }
}

int main() {
   
    string dir_output = "results/";
    createDirectory(dir_output);

   // 1) FRAMES LOADING - TO BE DELETED

    vector<cv::Mat> frames; //frames vector definition
    for (int i = 1; i <= 40; i++) {
        stringstream ss;
        ss << "input/frame_" << setfill('0') << setw(2) << i << ".png"; //<iomanip> is used just here; this program reads directly from a directory called 'input'
        cv::Mat img = cv::imread(ss.str(), 0); 
        if (img.empty()) {
            cout << "Errore: file not found " << ss.str() << endl;
            return -1;
        }
        frames.push_back(img);
    }

    // 2) GRADIENT SETTING (PREWITT OPERATOR) - I tried all derivative operators and Prewitt was the best one for me

    int height = frames[0].rows; //frame height
    int width = frames[0].cols; //frame width

    // Prewitt convolution kernels setting for gradient computation
    float kx[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}}; //kernel matrix operating in the x-direction
    float ky[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}}; //kernel matrix operating in the y-direction
    cv::Mat kernel_x(3, 3, CV_32FC1, kx); //cv::Mat objects creation (previous matrices)
    cv::Mat kernel_y(3, 3, CV_32FC1, ky);

    int gradient_thresh = 40; //a parameter (threshold level) for tuning performances of gradient operator

    // 3) FRAMES ELABORATION

    for (int i = 0; i < frames.size(); ++i) {
        string frame_num = cv::format("%02d", (i + 1)); //generating frames number

        // 3.1) SMOOTHING 
        cv::Mat smoothed_image;
        cv::GaussianBlur(frames[i], smoothed_image, cv::Size(3, 3), 0);

        // 3.2) GRADIENT (PREWITT) COMPUTATION
        cv::Mat grad_x, grad_y, abs_x, abs_y, grad_mag; //auxiliary matrices (variables): grad_x, grad_y are gradient values; abs_x, abs_y are their absolute values; grad_mag is the gradient's magnitude
        cv::filter2D(smoothed_image, grad_x, CV_16S, kernel_x); //derivative convolution finding vertical edges
        cv::filter2D(smoothed_image, grad_y, CV_16S, kernel_y); //derivative convolution finding horizontal edges
        
        cv::convertScaleAbs(grad_x, abs_x); //absolute values computation
        cv::convertScaleAbs(grad_y, abs_y);
        cv::addWeighted(abs_x, 0.5, abs_y, 0.5, 0, grad_mag); //weighted sum of vertical and horizontal edges (they have the same importance), computing grad_mag

        // 3.4) THRESHOLDING
        cv::Mat grad_thresholded; //the image after gradient and threshold computation
        cv::threshold(grad_mag, grad_thresholded, gradient_thresh, 255, cv::THRESH_BINARY); //computing a binary mask
        /* Eventually the threshold can be replaced by Otsu's with this code:
        cv::Mat grad_thresholded; //the image after gradient and threshold computation
        cv::threshold(grad_mag, grad_thresholded, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU); //computing a binary mask through Otsu's optimal thresholding
        */

        // 3.5) HOUGH TRANSFORM
        vector<cv::Vec4i> lines; //a vector containing all found lines
        int threshold_hough = 20;  //minimum number of lined up points to find a line through them
        double minLineLength = 15;  //minimum length of a valid line
        double maxLineGap = 5;    //maximum gap between two lines to be unified in a unique line
        
        cv::HoughLinesP(grad_thresholded, lines, 1, CV_PI / 180, threshold_hough, minLineLength, maxLineGap); //Hough function
        /*parameters meaning: the image after gradient and threshold, vector storing lines, "rho", "theta", 3 parameters just defined*/  

        for (int l = 0; l < lines.size(); l++) { //layering found lines (in green) on the gradient and threshold image
            cv::Vec4i ln = lines[l];
            cv::line(grad_thresholded, cv::Point(ln[0], ln[1]), cv::Point(ln[2], ln[3]), cv::Scalar(255), 1, cv::LINE_8);
        }

        // 3.6) FLOOR LINE AND OTHER NOISY ELEMENTS REMOVING
        // Creating a mask operating only on lower part of the image (from the middle to the bottom)
        cv::Mat floor = cv::Mat::zeros(height, width, CV_8UC1);
        cv::Rect low(0, height * 0.50, width, height * 0.50); //a mid-lower rectangle
        grad_thresholded(low).copyTo(floor(low)); //copy and past pixels from the lower rectangle of grad_thresholded to floor

        // Looking for horizontal lines (floor) to isolate, on the lower part of the image
        cv::Mat horiz_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(12, 1));//defining a rectangular structuring element
        cv::Mat horiz_lines;
        cv::morphologyEx(floor, horiz_lines, cv::MORPH_OPEN, horiz_kernel); //performing morphological opening on the lower part of the image
        
        // 3.6.A) Deleting found horizontal lines saving useful horizontal lines (e.g. arms in boxing sequences)
        cv::Mat nofloor;
        cv::bitwise_and(grad_thresholded, ~horiz_lines, nofloor); //inverse map on lower part of the image. Floor and noisy horizontal lines become black, being white
                                                                  //useful horizontal elements in lower part become white, being black              

        // 3.6.B) Morphological opening
        cv::Mat op = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::morphologyEx(nofloor, nofloor, cv::MORPH_OPEN, op);

        // 3.6.C) Horizontal morphological closing to fill the chest
        cv::Mat kernel_fill_h = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 1));
        cv::morphologyEx(nofloor, nofloor, cv::MORPH_CLOSE, kernel_fill_h);

        // 3.6.D) Vertical morphological closing for ankles and foot
        cv::Mat kernel_fill_v = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 9));
        cv::morphologyEx(nofloor, nofloor, cv::MORPH_CLOSE, kernel_fill_v);

        // 3.6.E) Closing silhouette
        cv::Mat kernel_cross = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
        cv::morphologyEx(nofloor, nofloor, cv::MORPH_CLOSE, kernel_cross);

        // 4) STORING OUTPUT
        string output_path = dir_output + "/grad_thresh_" + frame_num + ".png";
        cv::imwrite(output_path, nofloor);
    }

    cout << "\nOutput stored in '" << dir_output << "'." << endl;
    return 0;
}
