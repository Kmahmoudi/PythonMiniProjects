import cv2
from mtcnn.mtcnn import MTCNN

def count_and_tag_people_hog(image_path, image):
    # Create a HOG face detector
    hog_face_detector = cv2.HOGDescriptor()
    hog_face_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detect faces in the image using HOG
    faces_hog, _ = hog_face_detector.detectMultiScale(image)

    # Count the number of detected faces using HOG
    num_people_hog = len(faces_hog)

    # Draw bounding boxes around detected faces using HOG (in red)
    for (x, y, w, h) in faces_hog:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return num_people_hog

def count_and_tag_people_mtcnn(image_path, image):
    # Create an MTCNN detector
    detector = MTCNN()

    # Detect faces in the image using MTCNN
    faces_mtcnn = detector.detect_faces(image)

    # Count the number of detected faces using MTCNN
    num_people_mtcnn = len(faces_mtcnn)

    # Draw bounding boxes around detected faces using MTCNN (in blue)
    for face in faces_mtcnn:
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return num_people_mtcnn

# Example usage
image_path = 'C:/Users/Administrator/Pictures/people.jpg'

# Load the image
image = cv2.imread(image_path)

# Resize the image to fit on a small screen
small_screen_height = 600  # Set the desired height for the small screen
resized_image = cv2.resize(image, (int(small_screen_height * image.shape[1] / image.shape[0]), small_screen_height))

# Count and tag faces using HOG (in red)
num_people_hog = count_and_tag_people_hog(image_path, resized_image)
print(f"Number of people in the photo (HOG): {num_people_hog}")

# Count and tag faces using MTCNN (in blue)
num_people_mtcnn = count_and_tag_people_mtcnn(image_path, resized_image)
print(f"Number of people in the photo (MTCNN): {num_people_mtcnn}")

# Display the resized image with bounding boxes drawn around faces
cv2.imshow('Resized Image with HOG and MTCNN', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



