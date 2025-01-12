import cv2
import numpy as np
import mysql.connector
from tensorflow.keras.models import load_model
import time

# Load the model
model = load_model(r"C:/Users/Debaki/Desktop/intern/codes/attendance_rec/keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="bishal@11",
    database="attendance_system"
)
c = conn.cursor()

# Function to record attendance in the database
def record_attendance(name):
    try:
        # Check if the person is already in the database
        c.execute("SELECT * FROM attendance WHERE name = %s", (name,))
        result = c.fetchone()

        if result:
            # If the person exists, increment their attendance count
            c.execute("UPDATE attendance SET count = count + 1 WHERE name = %s", (name,))
            conn.commit()
            print(f"Attendance incremented for {name}")
        else:
            # If the person does not exist, add them to the database with count = 1
            c.execute("INSERT INTO attendance (name, count) VALUES (%s, %s)", (name, 1))
            conn.commit()
            print(f"Attendance marked for {name} with count = 1")
    except Exception as e:
        print(f"Error updating attendance: {e}")

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)

# Variables to control output
confidence_threshold = 0.95
cooldown_time = 3  # Cooldown time in seconds before allowing a new recognition
last_recognition_time = {}  # Dictionary to track last recognition time for each person

while True:
    # Grab the web camera's image
    ret, image = camera.read()

    # Resize the raw image into (224-height, 224-width) pixels
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image_resized)

    # Make the image a numpy array and reshape it to the model's input shape
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_array = (image_array / 127.5) - 1

    # Predict the model
    prediction = model.predict(image_array, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Clean the label
    confidence_score = prediction[0][index]

    current_time = time.time()

    # If the confidence is greater than the threshold
    if confidence_score >= confidence_threshold:
        last_time = last_recognition_time.get(class_name, 0)  # Get last recognition time for this person

        if current_time - last_time >= cooldown_time:
            print(f"Recognized as: {class_name} with confidence {confidence_score*100:.2f}%")
            record_attendance(class_name)  # Mark attendance in the database
            last_recognition_time[class_name] = current_time  # Update the last recognition time for this person

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the ESC key on your keyboard.
    if keyboard_input == 27:
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()

# Close the database connection
conn.close()
