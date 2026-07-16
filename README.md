# Human Action Recognition for Human-Robot Interaction (HRI)

## 1. Project Overview
In collaborative robotics and smart environments, computer vision systems are essential for understanding human behavior to ensure operational safety and provide contextual assistance [1]. This project develops a robust system capable of analyzing temporal image sequences to detect and classify six fundamental human actions:
*   **Walking**
*   **Jogging**
*   **Running**
*   **Boxing**
*   **Hand Waving**
*   **Hand Clapping**

## 2. Technical Pipeline
The system processes continuous streams of **40 pre-extracted frames** (approx. 1.6s of execution) to capture at least one complete kinematic cycle. The core pipeline includes:

1.  **Robust Segmentation:** Isolating the human actor from static background clutter.
2.  **Dynamic Tracking:** Following the subject's bounding box across the sequence.
3.  **Feature Extraction:** Analyzing spatial-temporal movement dynamics.
4.  **Global Classification:** Assigning a single action label to the entire 40-frame sequence.

## 3. Development Stack & Environment
To ensure absolute compatibility and performance, the project follows these technical constraints:
*   **Language:** C++ (Intermediate level required).
*   **Main Library:** **OpenCV** for pixel manipulation, color space management, and CV algorithms.
*   **Hybrid Layer:** Integration of lightweight pre-trained models (e.g., **MediaPipe**) is permitted solely for silhouette extraction stability under severe scale or clothing variations.
*   **Platform:** Developed and tested within the **Taliercio.2020 / VLAB Virtual Machine**.

## 4. Ambiguity Resolution Strategies
The framework methodically addresses two key motion recognition challenges:
*   **Motion Directionality (Clapping vs. Waving):** Resolves actions with identical silhouettes by isolating local motion vectors (horizontal convergence vs. vertical/angular oscillations).
*   **Kinematic Intensity (Jogging vs. Running):** Distinguishes identical mechanics by measuring **Centroid Velocity** (global translation magnitude) and **Stride Frequency** (temporal frequency of silhouette deformation).

## 5. Dataset & Evaluation
The system is validated using a stratified subset of the **KTH Action Recognition Dataset**.
*   **Structure:** 72 balanced action sequences (12 per class) across 4 environmental scenarios: standard outdoors (*d1*), scale/zoom variations (*d2*), clothing variations (*d3*), and variable indoor lighting (*d4*).
*   **Metrics:** 
    *   **Localization:** Intersection over Union (IoU) and mean IoU (mIoU) calculated on the median (20th) frame.
    *   **Classification:** Global Accuracy, F1-Score per class, and a mandatory Confusion Matrix.

## 6. How to Run
*(Note: Instructions should be adapted to your specific executable name)*
1. Access the **Taliercio.2020 VM** via VMWare.
2. Ensure **OpenCV** is correctly linked in your project settings.
3. Compile the C++ source code using the provided IDE/Compiler.
4. Execute the system providing the path to the 40-frame sequence folders.
