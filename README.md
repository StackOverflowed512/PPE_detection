# Personnel Detection and PPE Monitoring System

This project is a desktop application for personnel detection, face recognition, and PPE (Personal Protective Equipment) monitoring using YOLO and face_recognition. It provides a GUI for registering, editing, and monitoring personnel, and supports both live camera and video file input.

## Features

-   **Personnel Registration:** Register new people with name, age, function, ID, hashcode, and profile image.
-   **Edit Person:** Edit existing personnel information in the database via a user-friendly GUI.
-   **Face Recognition:** Detect and recognize registered personnel in real-time using a webcam or video file.
-   **PPE Detection:** Detect PPE (e.g., hardhat, mask, vest) using a YOLOv8 model.
-   **Info Panel:** Display detailed info and profile image on hover.
-   **Status Bar & HUD:** Show system status and camera info overlay.
-   **Database Storage:** All personnel data and face encodings are stored in a local SQLite database.

## Requirements

-   Python 3.8+
-   See [`requirements.txt`](requirements.txt) for all dependencies.

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/StackOverflowed512/PPE_detection
    cd detection
    ```

2. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Download/Place YOLOv8 PPE model:**

    - Place your trained YOLOv8 PPE model at the path specified in `config.json` (default: `models/ppe_custom_yolov8.pt`).

4. **Prepare configuration:**
    - Edit `config.json` as needed for your camera, model path, and display preferences.

## Usage

### Start the Application

```sh
python main_app.py
```

-   On launch, choose to **Register** a new person or **Detect** (start monitoring).
-   Use the GUI to register or edit personnel.
-   During detection, press `q` to quit the video window.

### Registering a Person

-   Click "Register New Person" in the GUI.
-   Fill in all required fields and select a profile image.
-   The face encoding will be extracted and stored in the database.

### Editing a Person

-   Click "Edit Person" in the GUI.
-   Select a person from the dropdown, edit their info, and click "Update".

### Detection

-   Click "Start Detection" in the GUI.
-   The system will recognize faces and detect PPE in real-time.
-   Hover over a face to see detailed info.

## File Structure

-   `main_app.py` - Main application and GUI logic.
-   `requirements.txt` - Python dependencies.
-   `config.json` - Application configuration.
-   `known_persons_data.json` - (Legacy) Example data format.
-   `registered_images/` - Folder for storing profile images.
-   `personnel_data.db` - SQLite database for personnel info.
-   `models/` - Folder for YOLOv8 model files.
-   `detection.ipynb` - Notebook for training PPE detection models.

## Notes

-   Make sure your webcam is connected or set the correct video source in `config.json`.
-   For best results, use clear, front-facing images for registration.
-   The application uses the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) library for PPE detection.
