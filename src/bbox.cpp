
/** 
* @brief    Bounding box drawing algorithm
* @details  Starting from a segmented frame, it performs the connected-components analysis in order to find the greater "blob's area" (the actor); then it ignores smaller noisy white areas and 
            verifies that greater white ones respect average human aspect ratio (height is greater than width); finally it draws a bounding box around the actor silhouette in the original frame.
* @author   Matteo Longo 
*/


#include "Bbox.h"

cv::Rect Bbox::bbox_func(const cv::Mat& input, cv::Mat& output){ 
    
    cv::Mat labels, stats, centroids;

    /** @brief the following "connectedComponentsWithStats" function is an OpenCV native function that detects connected components looking in all 8 neighbours of a pixel (in 8 directions)
    *@param input the segmented frame
    *@param labels a matrix containing the reference to the connected component belonging to
    *@param stats a matrix containing geometrical informations about each connected component (coordinates of leftmost corner pixel, height, width, area)
    *@param centroids a matrix containing ceontroid coordinates of each connected component
    *@param 8 directions connectivity for a pixel
    *@param CV_32S signed 32-bit integer output labels values
    */
    int num_labels = cv::connectedComponentsWithStats(input, labels, stats, centroids, 8, CV_32S);    //number of found connected components

    cv::Rect actor_bbox(0, 0, 0, 0);                                        //initialize empty bounding box
    cv::Mat finalSilhouette = cv::Mat::zeros(input.size(), CV_8UC1);        //initialize black image    

    bool actor_found = false;     

    int largest_label = -1; //the number identifying the greatest connected component so far 
    int max_area = 0;       //maximum area of the greatest connected component so far
    int min_area = 30;      //minimum area a connected component should have (ignore connected components having smaller area than 20)

    for (int j = 1; j < num_labels; j++) {              //looking for greatest connected component area (j=0 corresponds to no area)
        int area = stats.at<int>(j, cv::CC_STAT_AREA);  //area = value stored in the field having coordinates (j, column "area")
        int w = stats.at<int>(j, cv::CC_STAT_WIDTH);    //width w = value stored in the field having coordinates (j, cloumn "width")
        int h = stats.at<int>(j, cv::CC_STAT_HEIGHT);   //height h = value stored in the field having coordinates (j, column "height")

        /*
        computes the bounding box area and "compares" it to the area of the candidate actor, in order to exclude shadows having an "aspect ratio" similar to a human aspect ratio (h>w)
        */
        double bbox_area = static_cast<double>(w*h);                    //double conversion is necessary to avoid "zero results"
        double areas_ratio = static_cast<double>(area) / bbox_area;     //the ratio between the blob area (candidate actor) and the area of the virtual bounding box surrounding it

        if(areas_ratio >= 0.20) {                           
            if (area > min_area && area > max_area) {       //if current area > 20 pixel and greater than the greatest area found so far, update max_area with the current value
                max_area = area;
                largest_label = j;                          //store also its reference
            }
        }
    }

    if (largest_label != -1) {                                              //if an actor is found (good connected component)
        int x = stats.at<int>(largest_label, cv::CC_STAT_LEFT);
        int y = stats.at<int>(largest_label, cv::CC_STAT_TOP);
        int w = stats.at<int>(largest_label, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(largest_label, cv::CC_STAT_HEIGHT);
        actor_bbox = cv::Rect(x, y, w, h);                                  //actor's bounding box is delimited by that connected component boundaries

        finalSilhouette = cv::Mat::zeros(input.size(), CV_8UC1);            //initialize black image

        for (int r = 0; r < input.rows; r++) {                              //scan segmented image
           for (int c = 0; c < input.cols; c++) {                      
               if (labels.at<int>(r, c) == largest_label) {                 //if labels pixel is a pixel of the actor
                    finalSilhouette.at<uchar>(r, c) = 255;                  //fill it in white colour, else it is black
                }
            }
        }

        bool aspect_ratio = (h >= 20 && (h * 1.5) > w);                     //setting human aspect ratio parameters (height greater than 20 pixel and height greater than width; they may be modified for tuning)
        int frame_area = input.rows * input.cols;
        bool big_enough = (max_area > 250 && max_area < frame_area * 0.4);

        if (aspect_ratio && big_enough) {                                   //if aspect ratio is respected
            actor_found = true;                                             //an actor was found
        } 
        else {
            finalSilhouette = cv::Mat::zeros(input.size(), CV_8UC1);        //if aspect ratio is not respected, delete silhouette in the segmented frame
        }
    }

   if (actor_found) {
        cv::rectangle(output, actor_bbox, cv::Scalar(0, 0, 255), 2);        //draw a red bounding box on 'output' (the original frame) around the actor
        return actor_bbox;                                                  //this bounding box is useful for next phases (classification)
    }
    return cv::Rect(0, 0, 0, 0);                                            //if no actor is found return no bounding box

}