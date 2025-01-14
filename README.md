# Attendance Record System Using Face Recognition

This project implements an **Attendance Record System** that uses face recognition. The system employs a machine learning model trained using **Google Teachable Machine** and records attendance in a MySQL database.

## Project Description
This project captures images using a webcam, detects faces using a trained model, and records attendance in a database. It ensures that each person is recognized with a certain confidence level and prevents duplicate entries using a cooldown mechanism.

The main steps include:
1. **Face Detection**: Images are captured using OpenCV.
2. **Recognition**: A pre-trained model identifies individuals.
3. **Attendance Recording**: Attendance is marked in a MySQL database.

---

## Features
- Face recognition using a model trained on Google Teachable Machine.
- Real-time face detection and classification using OpenCV.
- MySQL database integration for storing and updating attendance records.
- Cooldown mechanism to avoid duplicate recognition.
- Configurable confidence threshold.

---

## Technologies Used
- **Python**: Programming language for implementing the system.
- **OpenCV**: For capturing and processing webcam images.
- **TensorFlow/Keras**: For loading and using the trained face recognition model.
- **MySQL**: For storing attendance records.
- **Google Teachable Machine**: For training the face recognition model.

---
## Training the model:
Training a model using Google's Teachable Machine is a straightforward process that allows you to create machine learning models without extensive coding knowledge. Here's how you can get started:

Step 1: Access Teachable Machine

Open your browser and visit Teachable Machine.
![image](https://github.com/user-attachments/assets/3d531afa-83f4-4b11-9bc1-23937984934d)

Step 2: Choose a Project Type

Click "Get Started".
![image](https://github.com/user-attachments/assets/36c9d313-85bd-4e83-9ae6-f4cd16f33d99)

Select Image Project since you’re working with face recognition.

Step 3: Add Classes
You’ll see two default classes (Class 1, Class 2). These represent the categories your model will classify.

Rename the classes to meaningful names like:

Your Name (e.g., "John")


Another Person (e.g., "Alice")

Background (optional, for cases where no face is detected).

Add more classes by clicking "Add a Class" if needed.
![image](https://github.com/user-attachments/assets/de4ec51d-7c60-483c-9fbb-ccae466001d6)

Step 4: Collect Images for Each Class
Use your webcam or upload images for each class:

Using Webcam:

Click "Webcam" under the desired class.

Collect images of yourself or another person by holding the button.

Take images from multiple angles (front, left, right) and in different lighting conditions.

Step 5: Train the Model

Once images for all classes are collected, click "Train Model".

The system will train the model using your provided data. This usually takes a few seconds to a couple of minutes, depending on the number of images.
![image](https://github.com/user-attachments/assets/fb1fb6c2-677e-4028-8df5-062a74c116c4)

Step 6: Test the Model

Use the Preview Webcam option to test the model in real-time.

Ensure the model correctly identifies faces for each class.

If the accuracy is low:

Add more training images, especially for poorly performing classes.

Include diverse conditions (e.g., different lighting, angles).
![image](https://github.com/user-attachments/assets/048b8598-f4ae-4619-b85e-b48af0701dda)

Step 7: Export the Model
Once you’re satisfied with the accuracy:
Click "Export Model".
![image](https://github.com/user-attachments/assets/3f73fed0-a278-4422-9e85-3c86c8fd831e)

Choose the TensorFlow tab (for integration with Python).

Download the following files:
![image](https://github.com/user-attachments/assets/6914d897-de04-4c0a-9537-9005d7cfb9b7)

keras_model.h5: The trained model file.

labels.txt: Contains the class names.

Step 8: Integrate the Model into Your Code
Place keras_model.h5 and labels.txt in your project directory.
Modify the code to load the model and use it for face recognition.
![image](https://github.com/user-attachments/assets/43b86731-3b81-4b10-a47f-3a2f7dbfa0d9)

### Tips for Better Accuracy
Lighting: Train the model with images in varied lighting conditions.
Angles: Capture images from different angles for better generalization.
Diverse Expressions: Include a range of facial expressions.
Class Balance: Ensure all classes have a similar number of images to avoid bias.
## Installation

### Prerequisites
1. Python 3.7 or higher
2. MySQL database
3. Required Python libraries:
   - `opencv-python`
   - `numpy`
   - `tensorflow`
   - `mysql-connector-python`

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/username/attendance-face-recognition.git
   cd attendance-face-recognition

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Create and configure the MySQL database:
   - Create a database named `attendance_system`.
   - Use the following SQL to create a table:

     ```sql
     CREATE TABLE attendance (
         id INT AUTO_INCREMENT PRIMARY KEY,
         name VARCHAR(255) NOT NULL,
         count INT DEFAULT 0
     );
     ```
![image](https://github.com/user-attachments/assets/f73e9a2e-feff-44b7-8abc-e176f8d1bde7)

4. Place the following files in the project directory:
   - `keras_model.h5` (Trained model file)
   - `labels.txt` (Labels corresponding to the trained model)

## Usage
1.Run the script:
```bash
python attendance_system.py
```
2.The system will:

•Open a webcam window to detect and recognize faces.

•Record attendance in the database.

3.Press the ESC key to exit the program.

## Database Structure

Database Name: attendance_system

Table Name: attendance

Columns:

id: Auto-incrementing unique ID for each record.

name: Name of the person recognized.

count: Number of times the person has been recognized.

## Code Explaination
### 1. Importing Libraries
```python
import cv2
import numpy as np
import mysql.connector
from tensorflow.keras.models import load_model
import time
```
cv2:

OpenCV for capturing and processing images from the webcam.

numpy: 

For array manipulation and numerical operations.

mysql.connector:

To connect and interact with a MySQL database.

tensorflow.keras.models.load_model:

For loading the trained face recognition model.

time:

For managing cooldown times between recognitions.

### 2. Load the Trained Model
```python
model = load_model(r"C:/Users/Debaki/Desktop/intern/codes/attendance_rec/keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
```
model: 

Loads the pre-trained model created using Google Teachable Machine.

class_names: 

Reads the labels associated with the model's classes from labels.txt.

### 3. Connect to MySQL Database
```python
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Pass",
    database="attendance_system"
)
c = conn.cursor()
```
conn: 

Establishes a connection to the MySQL database.

c: Creates a cursor to execute SQL queries.

### 4. Function to Record Attendance
```python
def record_attendance(name):
    try:
        c.execute("SELECT * FROM attendance WHERE name = %s", (name,))
        result = c.fetchone()
        if result:
            c.execute("UPDATE attendance SET count = count + 1 WHERE name = %s", (name,))
            conn.commit()
            print(f"Attendance incremented for {name}")
        else:
            c.execute("INSERT INTO attendance (name, count) VALUES (%s, %s)", (name, 1))
            conn.commit()
            print(f"Attendance marked for {name} with count = 1")
    except Exception as e:
        print(f"Error updating attendance: {e}")
```
record_attendance(name):

Updates the attendance database.

If the name exists:

Increments the attendance count by 1.

If the name does not exist:

Inserts a new record with an initial count of 1.

conn.commit():

Ensures the changes are saved to the database.

Error Handling:

Catches and logs any database errors.

### 5. Initialize Webcam
```python
camera = cv2.VideoCapture(0)
confidence_threshold = 0.95
cooldown_time = 3
last_recognition_time = {}
```
camera: 

Opens the default webcam (0). Change to 1 for an external webcam.

confidence_threshold:

Minimum confidence level to consider a recognition valid.

cooldown_time:

Prevents repeated attendance for the same person within a short period (3 seconds).

last_recognition_time: 

Tracks the last time a person was recognized.

### 6. Main Loop for Face Recognition
```python
while True:
    ret, image = camera.read()
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Image", image_resized)
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1
```
camera.read():

Captures an image frame from the webcam.

cv2.resize():

Resizes the frame to 224x224 pixels, the required input size for the model.

cv2.imshow(): 

Displays the resized frame in a window titled "Webcam Image."

image_array:

Converts the image into a NumPy array and normalizes pixel values between -1 and 1 to match the model's input format.

### 7. Model Prediction
```python
prediction = model.predict(image_array, verbose=0)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]
```
model.predict(image_array):

Passes the preprocessed image to the model for prediction.

np.argmax(prediction): 

Finds the class index with the highest confidence score.

class_name: 

Retrieves the corresponding label from labels.txt.

confidence_score:

The confidence of the model for the predicted class.

### 8. Record Attendance if Confidence is High
```python
if confidence_score >= confidence_threshold:
    last_time = last_recognition_time.get(class_name, 0)
    if current_time - last_time >= cooldown_time:
        print(f"Recognized as: {class_name} with confidence {confidence_score*100:.2f}%")
        record_attendance(class_name)
        last_recognition_time[class_name] = current_time
```
Checks if the confidence score meets the threshold.

Ensures the cooldown time has elapsed since the last recognition of the same person.

Calls record_attendance(class_name) to update the database.

### 9. Exit on ESC Key
```python
keyboard_input = cv2.waitKey(1)
if keyboard_input == 27:
    break
```
cv2.waitKey(1): 
Waits for a key press.

27:
ASCII code for the ESC key. Breaks the loop and stops the program.

### 10. Cleanup
```python
camera.release()
cv2.destroyAllWindows()
conn.close()
```
camera.release(): 

Releases the webcam.

cv2.destroyAllWindows():

Closes all OpenCV windows.

conn.close(): 

Closes the database connection.
![image](https://github.com/user-attachments/assets/71a2c29c-164b-4965-81a9-0f3ad77f13f2)
![image](https://github.com/user-attachments/assets/c494a7ea-457c-43e8-ac48-9ed3c3d6644b)

## Summary of Workflow
The webcam captures frames.
The frame is resized and preprocessed.
The trained model predicts the person's identity.
If the prediction confidence is high, the person's attendance is recorded in the MySQL database.
A cooldown mechanism prevents duplicate entries within a short period.
The program exits cleanly when the ESC key is pressed.
## Future Improvements
•Add a GUI for better user interaction.

•Support for multiple face recognition models.

•Integration with cloud services for remote attendance tracking.

•Enhanced security measures for database operations.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
