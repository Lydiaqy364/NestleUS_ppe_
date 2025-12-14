# PPE Detection and Person Tracking with YOLOv8

This repository contains YOLOv8-based scripts for Personal Protective Equipment (PPE) detection and person tracking. The workflow covers model training, testing, frame-level prediction, video processing, and person ID tracking.

The main implementation is provided in the notebook `ppe_yolov8.ipynb`.

## Main Features

- YOLOv8 model training on a custom PPE dataset
- Testing and visualization on test images
- Person tracking in videos with consistent ID assignment
- Frame-level prediction
- Video-to-frame conversion for data preparation

## File Overview

- `ppe_v1.ipynb`  
  Main notebook containing the full pipeline:
  - Model setup and training
  - Testing on the test image set
  - Person tracking with ID assignment in videos
  - Frame-level prediction
  - Video-to-frame conversion utilities

## Pipeline Overview

### 1. Set Up and Train
- Load YOLOv8 model
- Configure dataset and training parameters
- Train the model and save checkpoints

### 2. Test the Test Set Images
- Run inference on test images
- Visualize detection results
- Qualitatively evaluate PPE detection performance

### 3. Track with Person ID in Videos
- Perform detection and tracking on video inputs
- Assign unique IDs to each detected person
- Export annotated videos with tracking results

### 4. Predict Frames
- Run inference on individual image frames
- Save prediction outputs for further analysis

### 5. Convert Videos into Frames
- Extract frames from videos at a specified frame rate
- Generate image datasets for training or inference
