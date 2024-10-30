# Project Directory
### Code
1. **YOLO**

  - <a href="code/4A-VILLAFLOR-MIDTERMS-EXAM-YOLO.ipynb">YOLO TRAINING</a>

2. **SSD**

  - <a href="code/4A-VILLAFLOR-MIDTERMS-EXAM-SSD.ipynb">SSD TRAINING</a>



### Documentation
- <a href="documentation/4A-VILLAFLOR-MIDTERMS-EXAM-REPORT.md">Report</a>
- <a href="documentation/4A-VILLAFLOR-MIDTERMS-EXAM-report.pdf">Report (.pdf)</a>
### Video

- <a href="video/README.md">Video Presentation</a>

# Project Outline

### 1. Selection of Dataset and Algorithm:
- Each student will choose a dataset suitable for object detection tasks. The dataset can be from publicly available sources (e.g., COCO, PASCAL VOC) or one they create.
- Select an object detection algorithm to apply to the chosen dataset. Possible algorithms include:
  - **HOG-SVM (Histogram of Oriented Gradients with Support Vector Machine)**: A traditional method for object detection.
  - **YOLO (You Only Look Once)**: A real-time deep learning-based approach.
  - **SSD (Single Shot MultiBox Detector)**: A deep learning method balancing speed and accuracy.

## 2. Implementation:

### Data Preparation:
- Preprocess the dataset by resizing images, normalizing pixel values, and, if necessary, labeling bounding boxes for objects.

### Model Building:
- Implement the selected object detection algorithm using appropriate libraries (e.g., OpenCV for HOG-SVM, TensorFlow/Keras for YOLO or SSD).

### Training the Model:
- Use the training data to train the object detection model. For deep learning methods, fine-tune hyperparameters (e.g., learning rate, batch size, epochs) to optimize model performance.

### Testing:
- Evaluate the model on a test set to assess its detection capabilities.
- Ensure to capture edge cases where the model may struggle.


## 3. Evaluation:

### Performance Metrics:
- **Accuracy**: Overall success rate of object detection.
- **Precision**: The proportion of true positive detections out of all positive predictions.
- **Recall**: The proportion of true positive detections out of all actual positives in the dataset.
- **Speed**: Measure the time taken for the model to detect objects in an image or video frame.

### Comparison:
- Compare the results of the chosen model against other potential algorithms (e.g., how HOG-SVM compares to YOLO or SSD in terms of speed and accuracy).

## 4. Submission Instructions:

### Repository Setup:
- Create a folder named `Midterm-Project` within your GitHub repository (`CSST106-Perception-and-Computer-Vision`).

### File Organization:
- `code/`: Include all Python scripts or Jupyter notebooks.
- `images/`: Store processed images showing detection results.
- `documentation/`: Add the report (`report.md` or `report.pdf`).
- `video/`: Include the video file documenting the project (`video.mp4`).

### Filename Format:
- Use the format `[SECTION-LASTNAME-MP]` for files, e.g., `4D-Garcia-MP.py`, `4D-Garcia-MP-results.jpg`, `4D-Garcia-MP-video.mp4`.

### Deadline:
- Submit the project by the specified due date to avoid penalties for late submissions.