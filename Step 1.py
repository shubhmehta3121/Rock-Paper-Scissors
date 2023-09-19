import cv2
import sys
sys.path.append(r'C:\Users\SHUBH MEHTA\Documents\Pycharm Projects')
from Custom_Hands import custom as ch
import os
import shutil
import numpy as np
from cvzone.HandTrackingModule import HandDetector



# Define the labels for different image categories
labels = ['rock', 'paper', 'scissors', 'restart']

def gather(label, samples, save_path, start=False, image_size=400):
    """
    Capture images from the webcam for a specified label and save them in separate directories.

    Args:
        label (str): The current label for which images are being collected.
        samples (int): The number of samples/images to be collected for each label.
        save_path (str): The path where the collected images will be saved.
        start (bool, optional): Set to True to start capturing images. Defaults to False.
        image_size (int, optional): Size of the collected images. Defaults to 400.

    Returns:
        None
    """
    count = 0
    folder_path = os.path.join(save_path, label)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)

    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    padding = 20

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        hands,annotated_frames = detector.findHands(frame, flipType=False, draw=False)
        frame = ch(frame, hands, 4, 2, 4, 2,padding)

        if count == samples:
            break

        if not success:
            break

        if label == 'blank':
            # Create a white rectangle for the 'blank' label
            cv2.rectangle(frame, (350, 75), (550, 200), (255, 255, 255), 2)
            new = frame[75:200, 350:550]
            white = cv2.resize(new, (image_size, image_size))
        else:
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                crop = frame[y - padding:y + h + padding, x - padding:x + w + padding]
                white = np.ones((image_size, image_size, 3), np.uint8) * 255
                height, width = crop.shape[0], crop.shape[1]

                if height / width > 1:
                    new_height = image_size
                    new_width = int((image_size / height) * width)
                    new_image = cv2.resize(crop, (new_width, new_height))
                    gap = int((image_size - new_width) / 2)
                    white[:, gap:gap + new_width] = new_image
                else:
                    new_width = image_size
                    new_height = int((image_size / width) * height)
                    new_image = cv2.resize(crop, (new_width, new_height))
                    gap = int((image_size - new_height) / 2)
                    white[gap:gap + new_height, :] = new_image
                cv2.imshow('White',white)
        cv2.putText(frame, f"Collecting {count}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(f"Collecting images of {label}", frame)

        if start:
            cv2.imwrite(os.path.join(folder_path, f'{count}.jpg'), white)
            count += 1

        key = cv2.waitKey(1)
        if key == ord('c'):
            start = not start
        if key == ord('q'):
            break

    print(f"{count} image(s) of {label} saved to {folder_path}")
    cap.release()
    cv2.destroyAllWindows()

def main():
    for i in labels:
        gather(i, 500, 'Dataset')

if __name__ == "__main__":
    main()
