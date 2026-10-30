/**
 * @file performance_metrics.h
 * @brief Evaluation metrics for bounding box localization and action classification.
 * @author Ivan Arduini
 */

#ifndef PERFORMANCE_METRICS_H
#define PERFORMANCE_METRICS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @struct GroundTruthAnnotation
 * @brief Holds sequence ground-truth data taken from labels files.
 */
struct GroundTruthAnnotation {
    int class_id;            // Action class identifier (1 to 6)
    cv::Rect2f bounding_box; 
};

/**
 * @struct SequenceEvaluationRecord
 * @brief Stores per-sequence evaluation data and benchmark metrics.
 */
struct SequenceEvaluationRecord {
    std::string sequence_id;  
    std::string action_name;  
    int ground_truth_class;   // Actual class label (1 to 6)
    int predicted_class;      // Predicted class label (1 to 6)
    bool is_correct;          
    double iou_median_frame;  // Intersection over Union score evaluated at 20th frame
};



/**
 * @class PerformanceMetrics
 * @brief Computes localization metrics, exporting results to CSV and TXT.
 */
class PerformanceMetrics {
public:
    /**
     * @brief Constructor for the PerformanceMetrics class.
     */
    PerformanceMetrics() = default;

    /**
     * @brief Destructor for the PerformanceMetrics class.
     */
    ~PerformanceMetrics() = default;

    /**
     * @brief Loads ground truth data (action ID and 20th frame box) from a text file.
     * @param filepath Path to the annotation .txt file.
     * @param annotation Output struct where parsed data are saved.
     * @return True if read successfully, false otherwise.
     */
    static bool loadGroundTruth(const std::string& filepath, GroundTruthAnnotation& annotation);

    /**
     * @brief Saves the test results of a processed sequence.
     * @param record Struct containing the sequence metrics to store.
     */
    void addRecord(const SequenceEvaluationRecord& record);


    //=========================================
    //  LOCALIZATION METRICS (BOUNDING BOX)  //
    //=========================================
    /**
     * @brief Computes Intersection over Union (IoU) between two bounding boxes.
     * @param bboxA First bounding box.
     * @param bboxB Second bounding box.
     * @return double IoU score in range [0.0, 1.0].
     */
    static double computeIoU(const cv::Rect2f& bboxA, const cv::Rect2f& bboxB);

    /**
     * @brief Calculates Mean Intersection over Union (mIoU) across all recorded sequences.
     * @return double Average IoU score.
     */
    double calculateMeanIoU() const;


    //=========================================
    //   CLASSIFICATION METRICS (CATEGORY)   //
    //=========================================
    /**
     * @brief Computes the Global Accuracy of the system.
     * @return double The global accuracy percentage (value between 0.0 and 100.0).
     */
    double computeGlobalAccuracy() const;

    /**
     * @brief Computes the F1-Score for each action class.
     * @return std::vector<double> A vector containing the F1-Score for each class index.
     */
    std::vector<double> computeF1Scores() const;

    /**
     * @brief Computes the Confusion Matrix for the 6 action classes.
     * @details Builds a 6x6 matrix where rows represent ground truth classes 
     *          and columns represent predicted classes.
     * @return std::vector<std::vector<int>> The 6x6 confusion matrix.
     */
    std::vector<std::vector<int>> computeConfusionMatrix() const;



    //=========================================
    //          EXPORT FUNCTIONS             //
    //=========================================
    /**
     * @brief Exports all individual sequence evaluation records to a CSV file.
     * @param filepath Target output CSV destination path.
     * @return bool True on write success, false otherwise.
     */
    bool exportToCSV(const std::string& filepath) const;

    /**
     * @brief Generates and exports a report to a text file.
     * @param filepath Target output TXT destination path.
     * @return bool True on write success, false otherwise.
     */
    bool exportToTXT(const std::string& filepath) const;

   /**
     * @brief Automatically evaluates the entire dataset, aggregates records, and exports CSV and TXT reports.
     * @param csvFilePath Path where the output CSV report will be saved.
     * @param txtFilePath Path where the output TXT report will be saved.
     * @return bool True if evaluation and export succeed, false otherwise.
     */
    bool exportCompleteEvaluation(const std::string& csvFilePath, const std::string& txtFilePath);

    /**
     * @brief Clears all stored records.
     */
    void reset();

private:
    std::vector<SequenceEvaluationRecord> records; // Aggregated evaluation outcomes
};

#endif // PERFORMANCE_METRICS_H