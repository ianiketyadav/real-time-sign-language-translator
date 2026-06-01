# Real-Time Sign Language Translator

## Overview

Real-Time Sign Language Translator is a Computer Vision and Machine Learning based application that recognizes sign language gestures using a webcam and converts them into readable text in real time.

The system uses MediaPipe for hand landmark detection, TensorFlow/Keras for gesture classification, and OpenCV for real-time video processing and user interface rendering.

## Features

* Real-time webcam-based gesture recognition
* Hand landmark detection using MediaPipe
* Neural Network based gesture classification
* Confidence-based prediction filtering
* Stability-based character recognition
* Real-time text generation
* Custom OpenCV user interface

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* NumPy
* Scikit-Learn

## Project Workflow

1. Capture hand gestures through webcam
2. Extract hand landmarks using MediaPipe
3. Process landmark coordinates
4. Train gesture classification model
5. Predict gestures in real time
6. Convert predictions into readable text

## Files

* `main.py` → Real-time gesture recognition system
* `create_landmarks.py` → Landmark dataset generation
* `train_model.py` → Neural network model training
* `requirements.txt` → Project dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Project Report

Complete project documentation is available inside the `docs` folder.

## Author

Aniket Yadav
