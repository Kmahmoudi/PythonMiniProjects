import os
import cv2

def extract_and_save_faces(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Use Haarcascades for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for index, (x, y, w, h) in enumerate(faces):
        # Crop and save the face with a prefix and the original file name
        margin = int((10 / 100) * image.shape[0])
        y = max(0, y - margin)
        h = min(image.shape[0] - 1, h + 2 * margin)
        # Crop and save the face with a prefix and the original file name
        cropped_face = image[y:y+h, x:x+w]
        new_file_name = f"faceExtr_{os.path.basename(image_path).split('.')[0]}.jpg"
        cv2.imwrite(os.path.join('output_faces', new_file_name), cropped_face)
        print(f"Face saved: {new_file_name}")

# Specify the path to your images directory
images_path = 'images_path'

# Create an output directory for faces
output_path = 'output_faces'
os.makedirs(output_path, exist_ok=True)

# Iterate over image files and extract faces
for filename in os.listdir(images_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Adjust the file extensions accordingly
        image_path = os.path.join(images_path, filename)
        extract_and_save_faces(image_path)

print('Faces extraction completed.')
