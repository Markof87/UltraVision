/**
 * @file Sequencer.h
 * @brief Header file for the Sequencer class, which handles the sequencing of images.
 * @author Marco Fabiani
 */

#ifndef SEQUENCER_H
#define SEQUENCER_H

#include <opencv2/opencv.hpp>

/**
 * @class Sequencer
 * @brief The Sequencer class is responsible for managing the sequencing of images.
 * This class provides functionality to load, display, and navigate through a sequence of images.
 */

class Sequencer
{

public:
    /**
     * @brief Constructor for the Sequencer class.
     */
    Sequencer(int action_idx, int person_idx, int scenario_idx, bool reload, int play, int frame) : 
        current_action_idx(action_idx), current_person_idx(person_idx), current_scenario_idx(scenario_idx), need_reload(reload), play_state(play), frame_idx(frame) {};

    Sequencer() : current_action_idx(0), current_person_idx(0), current_scenario_idx(0), need_reload(true), play_state(0), frame_idx(0) {};

    /**
     * @brief Destructor for the Sequencer class.
     */
    virtual ~Sequencer() {};

    /**
     * @brief Runs the sequencer.
     * @details This is the main loop that runs the sequencer.
     * @param basePath The base path to the dataset.
     * @param windowName The name of the window to display.
     */
    void run(const std::string& basePath, const std::string& windowName); ///< Main loop to run the sequencer.

    //Getters and setters for private members
    /**
     * @brief Gets the index of the current action in the sequence.
     * @return The index of the current action.
     */
    int getCurrentActionIdx() const { return current_action_idx; }

    /**
     * @brief Sets the index of the current action in the sequence.
     * @param idx The index to set for the current action.
     */
    void setCurrentActionIdx(int idx) 
    { 
        if(current_action_idx != idx)
        {
            need_reload = true;
            current_action_idx = idx; 
        }

    }

    /**
     * @brief Gets the index of the current person in the sequence.
     * @return The index of the current person.
     */
    int getCurrentPersonIdx() const { return current_person_idx; }

    /**
     * @brief Sets the index of the current person in the sequence.
     * @param idx The index to set for the current person.
     */
    void setCurrentPersonIdx(int idx) 
    { 
        if(current_person_idx != idx)
        {
            need_reload = true;
            current_person_idx = idx; 
        }
    }

    /**
     * @brief Gets the index of the current scenario in the sequence.
     * @return The index of the current scenario.
     */
    int getCurrentScenarioIdx() const { return current_scenario_idx; }

    /**
     * @brief Sets the index of the current scenario in the sequence.
     * @param idx The index to set for the current scenario.
     */
    void setCurrentScenarioIdx(int idx) 
    { 
        if(current_scenario_idx != idx)
        {
            need_reload = true;
            current_scenario_idx = idx; 
        }
    }

    /**
     * @brief Checks if the sequence needs to be reloaded.
     * @return True if the sequence needs to be reloaded, false otherwise.
     */
    bool isNeedReload() const { return need_reload; }

    /**
     * @brief Sets the flag indicating whether the sequence needs to be reloaded.
     * @param reload True to indicate that the sequence needs to be reloaded, false otherwise.
     */
    void setNeedReload(bool reload) { need_reload = reload; }

    /**
     * @brief Gets the current play state of the sequencer.
     * @return The current play state (0: paused, 1: playing).
     */
    int getPlayState() const { return play_state; }

    /**
     * @brief Sets the current play state of the sequencer.
     * @param state The play state to set (0: paused, 1: playing).
     */
    void setPlayState(int state) { play_state = state; }

    /**
     * @brief Gets the index of the current frame in the sequence.
     * @return The index of the current frame.
     */
    int getFrameIdx() const { return frame_idx; }

    /**
     * @brief Sets the index of the current frame in the sequence.
     * @param idx The index to set for the current frame.
     */
    void setFrameIdx(int idx) { frame_idx = idx; }

private:

    /**
     * @brief Updates the loader with the current sequence based on the selected action, person, and scenario.
     * @param loadedSequence The sequence of images to be updated.
     * @param basePath The base path to the dataset.
     */
    void updateLoader(std::vector<cv::Mat>& loadedSequence, const std::string& basePath);

    /**
     * @brief Draws the Heads-Up Display (HUD) on the canvas.
     * @param canvas The canvas on which to draw the HUD.
     * @param currentFrame The current frame to display.
     * @param totalFrames The total number of frames in the sequence.
     */
    void drawHUD(cv::Mat& canvas, const cv::Mat& currentFrame, size_t totalFrames);

    /**
     * @brief Handles keyboard input.
     * @param key The key that was pressed.
     * @param totalFrames The total number of frames in the sequence.
     * @return True if the key was handled, false otherwise.
     */
    bool handleKeyboard(char key, size_t totalFrames);

    int current_action_idx; ///< Index of the current action in the sequence.
    int current_person_idx; ///< Index of the current person in the sequence.
    int current_scenario_idx; ///< Index of the current scenario in the sequence.

    bool need_reload; ///< Flag indicating whether the sequence needs to be reloaded.
    int play_state; ///< State of the sequencer (0: paused, 1: playing).
    int frame_idx; ///< Index of the current frame in the sequence.

};

#endif // SEQUENCER_H