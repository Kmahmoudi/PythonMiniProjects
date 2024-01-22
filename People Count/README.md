# Face Detection with HOG and MTCNN

This repository demonstrates face detection using two different methods: HOG (Histogram of Oriented Gradients) and MTCNN (Multi-task Cascaded Convolutional Networks). The provided Python code processes an image and draws bounding boxes around detected faces using both methods.

## HOG Face Detection

### Overview

HOG is a feature descriptor widely used for object detection. In the context of face detection, it analyzes the distribution of gradients in an image to identify faces.

### Code Explanation

1. **Load the image:**
   - The image is loaded using OpenCV's `cv2.imread` function.

2. **HOG Face Detection:**
   - The HOG face detection is implemented using `cv2.HOGDescriptor` with the default people detector.

3. **Count and Tag Faces:**
   - Bounding boxes are drawn around detected faces, and the number of people is counted.

4. **Display Result:**
   - The original image with bounding boxes around detected faces is displayed.

## MTCNN Face Detection

### Overview

MTCNN is a deep learning-based model specifically designed for face detection. It detects faces in multiple stages and can handle various face orientations.

### Code Explanation

1. **Load the image:**
   - The image is loaded using OpenCV's `cv2.imread` function.

2. **MTCNN Face Detection:**
   - The MTCNN face detection is implemented using the `mtcnn` library.

3. **Count and Tag Faces:**
   - Bounding boxes are drawn around detected faces, and the number of people is counted.

4. **Display Result:**
   - The original image with bounding boxes around detected faces is displayed.

## Usage

1. Install the required libraries:

   ```bash
   pip install opencv-python mtcnn
