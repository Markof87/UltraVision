#include "MainWindow.h"

#include <QMessageBox>
#include <QDir>
#include <QDebug>

/**
 * @file MainWindow.cpp
 * @brief Implementation of the MainWindow class.
 * @author Marco Fabiani
 */

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), currentFrameIndex(0), isPlaying(false)
{
    //Initialize the user interface and set the window title and size
    setupUi(this);
    setWindowTitle("UltraVision");
    setFixedSize(800, 600);

    //Timer to play (25 FPS -> 40ms)
    playTimer = new QTimer(this);
    connect(playTimer, &QTimer::timeout, this, &MainWindow::onNextFrame);
}

MainWindow::~MainWindow()
{
    //Clean up the timer
    if (playTimer) {
        playTimer->stop();
        delete playTimer;
    }
}