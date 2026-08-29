#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QStatusBar>
#include <QComboBox>

#include <opencv2/opencv.hpp>

/**
 * @class MainWindow
 * @brief The MainWindow class represents the main window of the application.
 * @author Marco Fabiani
 */

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    /**
     * @brief Slot to handle the play button click event.
     */
    void onLoadSequence();

    /**
     * @brief Slot to handle the pause button click event.
     */
    void onPlayPause();

    /**
     * @brief Slot to handle the next frame button click event.
     */
    void onNextFrame();

    /**
     * @brief Slot to handle the manual frame selection.
     * @param position The position of the slider representing the selected frame.
     */
    void onSliderMoved(int position);

private:
    /**
     * @brief Set up the user interface.
     */
    void setupUI();

    /**
     * @brief Load a frame from the video sequence.
     * @param index The index of the frame to load.
     */
    void loadFrame(int index);

    /**
     * @brief OpenCV utility conversion function to convert a cv::Mat to QImage.
     * @param mat The input image in cv::Mat format.
     * @return The converted QImage.
     */
    QImage cvMatToQImage(const cv::Mat& mat);

    //GUI elements
    QWidget *centralWidget;
    QLabel *imageLabel;
    QPushButton *loadButton;
    QPushButton *playPauseButton;
    QSlider *frameSlider;
    QLabel *infoLabel;
    QComboBox *actionSelector;

    //Control variables for OpenCV playing
    QTimer *playTimer;
    std::vector<cv::Mat> loadedFrames;
    int currentFrameIndex;
    bool isPlaying;
    QString currentSequencePath;
};

#endif // MAINWINDOW_H