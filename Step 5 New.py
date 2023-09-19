import cv2
import numpy as np
from random import choice
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import sys
import time
from time import time,sleep
sys.path.append(r'C:\Users\SHUBH MEHTA\Documents\Pycharm Projects')
from Custom_Hands import custom as ch

labels = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3:"restart"
}
padding = 20
image_size = 227

classifier = Classifier('TM MODEL/keras_model.h5', 'TM MODEL/labels.txt')

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)
cv2.namedWindow('Game')
#cv2.resizeWindow('Game', 1200, 800)
seconds = 5
stat_screen = 5
winner_screen = 5
game_started=False
stats = {'User': 0, 'Computer': 0, 'Tie': 0}

def draw_box(frame, color, length=50, thickness=2):
    """
    Draws a box with specified characteristics on the given frame.

    Args:
        frame (numpy.ndarray): The image frame on which the box will be drawn.
        color (tuple): The color of the box lines in (B, G, R) format.
        length (int, optional): The length of each line segment of the box. Default is 50.
        thickness (int, optional): The thickness of the box lines. Default is 2.
    """
    # Define corner points for the box
    x1, y1 = 320, 10
    x2, y2 = 630, 10
    x3, y3 = 320, 300
    x4, y4 = 630, 300

    # Draw lines to create the box
    cv2.line(frame, (x1, y1), (x1 + length, y1), color=color, thickness=thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color=color, thickness=thickness)

    cv2.line(frame, (x2, y2), (x2 - length, y2), color=color, thickness=thickness)
    cv2.line(frame, (x2, y2), (x2, y2 + length), color=color, thickness=thickness)

    cv2.line(frame, (x3, y3), (x3 + length, y3), color=color, thickness=thickness)
    cv2.line(frame, (x3, y3), (x3, y3 - length), color=color, thickness=thickness)

    cv2.line(frame, (x4, y4), (x4 - length, y4), color=color, thickness=thickness)
    cv2.line(frame, (x4, y4), (x4, y4 - length), color=color, thickness=thickness)

def calculate_winner(move1, move2):
    """
    Determines the winner between two moves in a rock-paper-scissors game.

    Args:
        move1 (str): The first player's move, can be "rock", "paper", or "scissors".
        move2 (str): The second player's move, can be "rock", "paper", or "scissors".

    Returns:
        str: The result of the game, can be "User" if the first player wins, "Computer" if the second player wins, or "Tie" if it's a tie.
    """
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"

while True:
    success,frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame,1)
    cv2.rectangle(frame, (10, 10), (300, 300), (255, 255, 255), 2)
    draw_box(frame, (0, 0, 255))
    hands,annotated_frame = detector.findHands(frame, flipType=False, draw=False)
    frame = ch(frame, hands, 4, 2, 4, 2, padding)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        crop = frame[y - padding:y + h + padding, x - padding:x + w + padding]
        white = np.ones((image_size, image_size, 3), np.uint8) * 255
        height, width = crop.shape[0], crop.shape[1]

        if crop.shape[0] > 0 and crop.shape[1] > 0:
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

        if (x >= 320) and (x + w <= 630) and (y >= 10) and (y + h <= 300):
            draw_box(frame, (0, 255, 0))
            icon = cv2.imread('gestures/choice.jpg')
            icon_resize = cv2.resize(icon, (290, 290))
            frame[10:300, 10:300] = icon_resize

            if not game_started:
                start_time = time()
                game_started = True
            else:
                elapsed_time = time() - start_time
                remaining_time = max(0, seconds - int(elapsed_time))
                cv2.putText(frame, f"Game starts in : {remaining_time}", (10, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                if remaining_time == 0:
                    pred, index = Classifier.getPrediction(classifier, white, draw=False)
                    user_choice = labels[np.argmax(pred)]
                    if user_choice=='restart':
                        cv2.namedWindow('Stats')
                        start_stat_time = time()
                        blank = np.zeros((512, 512, 3), np.uint8)
                        cv2.putText(blank, f'Total games: {stats["User"] + stats["Computer"] + stats["Tie"]} games',(50, 50),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(blank, f'Player won {stats["User"]} games', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 2)
                        cv2.putText(blank, f'Computer won {stats["Computer"]} games', (50, 250),cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 2)
                        cv2.putText(blank, f'There are {stats["Tie"]} ties', (50, 350), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255),2)

                        while int(time()-start_stat_time)<=stat_screen:
                            cv2.imshow('Stats', blank)
                            cv2.waitKey(1)
                        game_started=False
                        cv2.destroyWindow('Stats')

                    computer_choice = choice(['rock','paper','scissors'])  # Random computer choice
                    result = calculate_winner(user_choice, computer_choice)
                    if result in ['User','Computer','Tie']:
                        stats[result]+=1
                    winner_start_time = time()
                    cv2.namedWindow('Result')
                    #cv2.resizeWindow('Result',1200,1200)

                    while int(time()-winner_start_time)<winner_screen:
                        if result == 'User':
                            icon = cv2.imread('gestures/win.png')
                        elif result == 'Computer':
                            icon = cv2.imread('gestures/lose.jpg')
                        else:
                            icon = cv2.imread('gestures/tie.jpg')
                        icon = cv2.resize(icon, (310, 290))
                        result_blank = np.ones((480, 640, 3), np.uint8) * 125
                        result_icon = cv2.imread(f'gestures/{computer_choice}.png')
                        result_icon = cv2.resize(result_icon, (290, 290))
                        result_blank[10:300, 10:300] = result_icon
                        result_blank[10:300,320:630]=icon
                        cv2.putText(result_blank,f'New Game starts in {winner_screen-int(time()-winner_start_time)} seconds',(10,350),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
                        cv2.imshow('Result',result_blank)
                        cv2.imshow('Game',frame)
                        cv2.waitKey(1)
                        game_started=False
                    cv2.destroyWindow('Result')

        else:
            game_started=False
            icon = cv2.imread('gestures/blank.jpg')
            icon_resize = cv2.resize(icon, (290, 290))
            frame[10:300, 10:300] = icon_resize
            frame[10:300, 10:300] = icon_resize
            instruction = np.ones((512, 512, 3), np.uint8) * 255
            instruction = cv2.resize(instruction, (620, 150))
            frame[320:470, 10:630] = instruction
            cv2.putText(frame, 'Please Play', (10, 365), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
            cv2.putText(frame, '--> Place your hand in the box and make the gesture', (10, 400),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, '    for either "Rock","Paper","Scissors" (palm facing screen always)', (20, 415), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 255), 1)
            cv2.putText(frame, '--> Do a "Thumbs Up" for knowing your stats', (10, 445), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 255), 1)
            cv2.putText(frame, 'BY: SHUBH MEHTA', (505, 460), cv2.FONT_HERSHEY_COMPLEX, 0.40, (0, 0, 0), 1)
    else:
        game_started=False
        icon = cv2.imread('gestures/blank.jpg')
        icon_resize = cv2.resize(icon, (290, 290))
        frame[10:300, 10:300] = icon_resize
        frame[10:300, 10:300] = icon_resize
        instruction = np.ones((512, 512, 3), np.uint8) * 255
        instruction = cv2.resize(instruction, (620,150))
        frame[320:470, 10:630] = instruction
        cv2.putText(frame, 'Please Play', (10, 365), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 1)
        cv2.putText(frame, '--> Place your hand in the box and make the gesture', (10, 400), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 255), 1)
        cv2.putText(frame, '    for either "Rock","Paper","Scissors" (palm facing screen always)', (20, 415), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 255), 1)
        cv2.putText(frame, '--> Do a "Thumbs Up" for knowing your stats', (10, 445), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 255), 1)
        cv2.putText(frame, 'BY: SHUBH MEHTA', (505,460), cv2.FONT_HERSHEY_COMPLEX, 0.40, (0, 0, 0), 1)
    cv2.imshow('Game',frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
