# 📘 CSST106-4A: Machine Problem No. 1
## Exploring the Role of Computer Vision and Image Processing in AI

---

[Watch the video](assets/videos/Topic.1.1.Introduction.to.Computer.Vision.and.Image.Processing.mp4)


## 🎥 Introduction to Computer Vision and Image Processing

**Overview:**

Computer Vision is a branch of Artificial Intelligence (AI) that enables machines to interpret and understand visual information from the world. The primary goal is to replicate human visual capabilities, allowing machines to recognize objects, understand their environment, and make informed decisions based on visual inputs.

**Key Points:**

- **Role of Image Processing in AI:** Image processing is crucial for enhancing and analyzing images. It involves several stages:
  - **Preprocessing:** Improving image quality before analysis.
  - **Feature Extraction:** Identifying and extracting meaningful features from images.
  - **Decision-Making:** Utilizing deep learning to derive insights and make decisions.

---

## 🧩 Types of Image Processing Techniques

**1. Filtering:**

Filtering involves modifying an image to enhance or suppress certain features.

- **Examples:**
  - **Gaussian Blur:** Reduces noise and detail.
  - **Median Filter:** Removes noise by replacing pixel values with median values.
  - **Sharpening Filter:** Enhances image edges and details.

- **Applications in AI:**
  - **Noise Reduction:** Improves detection accuracy.
  - **Detail Enhancement:** Aids in object recognition and classification.

**2. Edge Detection:**

Edge detection identifies boundaries in images where intensity changes significantly.

- **Examples:**
  - **Canny Edge Detection:** Finds edges by detecting rapid intensity changes.
  - **Sobel Operator:** Computes image gradients to detect edges.
  - **Prewitt Operator:** Detects edges using gradient calculations.

- **Applications in AI:**
  - **Object Recognition:** Helps in outlining objects for easier classification.
  - **Feature Extraction:** Highlights important features for further analysis.

**3. Segmentation:**

Segmentation divides an image into distinct regions or objects for easier analysis.

- **Examples:**
  - **Thresholding:** Segments based on pixel intensity.
  - **Region Growing:** Expands regions based on pixel similarity.
  - **Watershed Algorithm:** Segments by identifying distinct regions using topological concepts.

- **Applications in AI:**
  - **Object Detection:** Isolates objects for detailed analysis.
  - **Medical Imaging:** Segments anatomical structures for diagnosis.

---

## 📊 Case Study Overview

**Selected AI Application:**

**Retail Inventory Management**

**How Image Processing is Used:**

- **Image Acquisition:** Captures images of store shelves using cameras.
- **Preprocessing:** Enhances image quality to ensure clear visibility of products.
- **Feature Extraction:** Identifies product types and quantities.
- **Segmentation:** Separates individual products from the background.

**Challenges Addressed:**

- **Accurate Inventory Tracking:** Ensures accurate counts and reduces manual errors.
- **Automated Restocking:** Helps in predicting inventory needs based on real-time analysis.

---

## 🔍 Your Image Processing Implementation

**Model:**

**Object Detection in Retail Environments**

**How It Works:**

- **Input:** Images of store shelves.
- **Processing:** Applies filtering and edge detection techniques.
- **Output:** Detects and labels products on the shelves.

**How It Helps:**

- **Improves Inventory Accuracy:** Provides precise counts of products.
- **Enhances Operational Efficiency:** Reduces the need for manual stock checks.

**Example Code Snippet:**

```python
import cv2
import numpy as np

# Load a pre-trained object detection model
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'model.caffemodel')

# Load an image from file (replace 'store_shelf.jpg' with your image)
image = cv2.imread('store_shelf.jpg')
height, width = image.shape[:2]

# Prepare the image for object detection
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
model.setInput(blob)
detections = model.forward()

# Process the detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:  # Threshold for detection confidence
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        (startX, startY, endX, endY) = box.astype("int")
        label = f"Object {i} ({confidence*100:.2f}%)"
        
        # Draw bounding box and label on the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![Example Image](assets/images/example.png)

## 🏁 Conclusion
**Summary:**

Effective image processing plays a pivotal role in the realm of artificial intelligence, serving as the foundation upon which many AI systems build their capabilities to interpret and analyze visual data. The application of image processing techniques such as filtering, edge detection, and segmentation significantly enhances the ability of AI systems to process visual information with greater accuracy and efficiency.

Filtering: This technique improves image quality by removing noise and enhancing important features. By smoothing out unwanted variations and emphasizing key details, filtering allows AI systems to work with cleaner and more reliable data, which is essential for accurate analysis and decision-making.

Edge Detection: Identifying boundaries within an image where there is a significant change in intensity or color is crucial for understanding the structure of visual elements. Edge detection helps in outlining objects and features within an image, making it easier for AI systems to recognize and classify them. This is particularly useful in applications such as object detection and facial recognition.

Segmentation: Segmenting an image into distinct regions or objects simplifies the analysis process. By isolating different parts of an image, segmentation enables AI systems to focus on specific areas of interest. This technique is vital for tasks like object detection, where precise identification and localization of items are required. It also plays a crucial role in medical imaging, where accurate segmentation of anatomical structures aids in diagnosis and treatment planning.

Together, these image processing techniques empower AI systems to achieve more precise object recognition, feature extraction, and overall decision-making. They enable machines to understand and interpret visual data in a way that closely mimics human visual perception, leading to more effective and intelligent applications across various domains.

In conclusion, the integration of robust image processing techniques into AI systems not only enhances their ability to handle complex visual data but also drives advancements in fields ranging from autonomous vehicles to medical diagnostics. The continual development and refinement of these techniques will play a critical role in the future evolution of AI and its applications.