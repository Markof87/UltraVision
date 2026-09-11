/*Performs the connected-components analysis in order to find greater "blob's area" (the actor) and to ignore smaller noisy
white areas, verifying that greater white areas respect average human aspect ratio (height is greater than width)*/

/** 
* @brief    Bounding box drawing algorithm
* @details  It draws a bounding box around the actor silhouette in the original frame.
* @author   Matteo Longo 
*/


#include <Bbox.h>

void bbox_func(const cv::Mat& input, const cv::Mat& output){ 
    
    cv::Mat labels, stats, centroids;

    /** @brief the following "connectedComponentsWithStats" function is an OpenCV native function that detects connected components looking in all 8 neighbours of a pixel (in 8 directions)
    *@param segmented the input segmented frame
    *@param labels a matrix containing the reference to the connected component belonging to
    *@param stats a matrix containing geometrical informations about each connected component (coordinates of leftmost corner pixel, height, width, area)
    *@param centroids a matrix containing ceontroid coordinates of each connected component
    *@param 8 directions connectivity for a pixel
    *@param CV_32S signed 32-bit integer output labels values
    */
    int num_labels = cv::connectedComponentsWithStats(segmented, labels, stats, centroids, 8, CV_32S);    //number of found connected components

    cv::Rect actor_bbox(0, 0, 0, 0);                                        //initialize empty bounding box
    cv::Mat finalSilhouette = cv::Mat::zeros(segmented.size(), CV_8UC1);    //initialize black image    

    bool actor_found = false; //boolean variable    

    int largest_label = -1; //the number identifying the gretest connected component so far 
    int max_area = 0;       //maximum area of the greatest connected component so far
    int min_area = 20;      //minimum area a connected component should have (ignore connected components having smaller area than 20)

    for (int j = 1; j < num_labels; j++) {              //looking for greatest connected component area (j=0 corresponds to no area)
        int area = stats.at<int>(j, cv::CC_STAT_AREA);  //area = value stored in the field having coordinates (j, column "area")
        if (area > min_area && area > max_area) {       //if current area > 20 pixel and greater than the greatest area found so far, update max_area with the current value
            max_area = area;
            largest_label = j;                          //store also its reference
        }
    }

    if (largest_label != -1) {                                              //if actor is found (good connected component)
    int x = stats.at<int>(largest_label, cv::CC_STAT_LEFT);
    int y = stats.at<int>(largest_label, cv::CC_STAT_TOP);
    int w = stats.at<int>(largest_label, cv::CC_STAT_WIDTH);
    int h = stats.at<int>(largest_label, cv::CC_STAT_HEIGHT);
    actor_bbox = cv::Rect(x, y, w, h);                                  //actor's bounding box is delimited by that connected component boundaries

    finalSilhouette = cv::Mat::zeros(segmented.size(), CV_8UC1);        //initialize black image

    for (int r = 0; r < segmented.rows; r++) {                          //scan segmented image
           for (int c = 0; c < segmented.cols; c++) {                      
               if (labels.at<int>(r, c) == largest_label) {                //if labels pixel is a pixel of the actor
                    filteredSilhouette.at<uchar>(r, c) = 255;              //fill it in white colour, else it is black
                }
            }
        }

        bool aspect_ratio = (h >= 20 && h > w);     //setting human aspect ratio parameters (height greater than 20 pixel and height greater than width; they may be modified for tuning the result)

        if (aspect_ratio) {                                                    // if aspect ratio is respected
            actor_found = true;                                                //an actor was found
        } 
        else {
            finalSilhouette = cv::Mat::zeros(segmented.size(), CV_8UC1);       //if aspect ratio is not respected, delete silhouette in the segmented frame
        }
    }

    cv::Mat bbox_frame;
    cv::cvtColor(frame, bbox_frame, cv::COLOR_GRAY2BGR);                       //temporally converts the grayscale original frame in a colored BGR image to draw a red bounding box

    if (actor_found) {
        cv::rectangle(bbox_frame, actor_bbox, cv::Scalar(0, 0, 255), 2);       //draw a red bounding box on bbox_frame around connected components boundaries
    }

}