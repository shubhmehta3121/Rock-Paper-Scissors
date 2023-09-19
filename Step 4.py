import cv2
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import sys
sys.path.append(r'C:\Users\SHUBH MEHTA\Documents\Pycharm Projects')
from Custom_Hands import custom as ch

# Define class labels for mapping class indices to names
labels = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3:"restart"
}

# Set padding and image size
padding = 20
image_size = 227

# Load the pre-trained model 'rock-paper-scissors-model.h5'
model = load_model("rock-paper-scissors-model.h5")

# Open the webcam using cv2.VideoCapture
cap = cv2.VideoCapture(0)

# Create a resizable window for displaying testing images
detector = HandDetector(maxHands=1)
cv2.namedWindow('Testing Images')
cv2.resizeWindow('Testing Images', 1200, 800)

while True:
    # Capture a frame from the webcam
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hands,annotated_frames = detector.findHands(frame, flipType=False, draw=False)
    frame = ch(frame, hands, 4, 2, 4, 2, padding)

    if not success:
        break

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        # Crop the hand region and create a white canvas
        crop = frame[y - padding:y + h + padding, x - padding:x + w + padding]
        white = np.ones((image_size, image_size, 3), np.uint8) * 255

        height, width = crop.shape[0], crop.shape[1]

        if height / width > 1:
            # Resize and center horizontally
            new_height = image_size
            new_width = int((image_size / height) * width)
            new_image = cv2.resize(crop, (new_width, new_height))
            gap = int((image_size - new_width) / 2)
            white[:, gap:gap + new_width] = new_image
            cv2.imshow('White', white)
        else:
            # Resize and center vertically
            new_width = image_size
            new_height = int((image_size / width) * height)
            new_image = cv2.resize(crop, (new_width, new_height))
            gap = int((image_size - new_height) / 2)
            white[gap:gap + new_height, :] = new_image
            cv2.imshow('White', white)
        pred = model.predict(np.array([white]))
        move_code = np.argmax(pred[0])
        move_name = labels[move_code]
    else:
        move_name = 'blank'

    # Display the predicted gesture name on the frame
    cv2.putText(frame, f'{move_name}', fontScale=1, fontFace=cv2.FONT_HERSHEY_COMPLEX, org=(50, 25), color=(0, 0, 255))
    cv2.imshow('Testing Images', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
