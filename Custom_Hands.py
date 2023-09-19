import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

colours = {'THUMB':(0,255,0),
           'INDEX':(255,0,0),
           'MIDDLE':(0,0,255),
           'RING':(255,255,0),
           'PINKY':(0,255,255),
           'PALM':(255,0,255)}

def custom(frame, hands,outer_radius,inner_radius,tips,thickness,padding):
    hand_landmarks = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
                      'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP',
                      'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP',
                      'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP',
                      'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
                      'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
    if hands:
        hand = hands[0]
        positions = {i:hand_landmarks[i] for i in range(21)}
        landmarks = {hand_landmarks[i]: hand['lmList'][i][:2] for i in range(21)}
        x,y,w,h = hand['bbox']
        cv2.rectangle(frame,(x-padding,y-padding),(x+w+padding,y+h+padding),(255,0,255),2)


        #For Thumb
        cv2.line(frame, landmarks[positions[1]],landmarks[positions[2]],colours['THUMB'],thickness)
        cv2.circle(frame, landmarks[positions[2]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[2]], inner_radius, colours['THUMB'], thickness)

        cv2.line(frame, landmarks[positions[2]],landmarks[positions[3]],colours['THUMB'],thickness)
        cv2.circle(frame, landmarks[positions[3]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[3]], inner_radius, colours['THUMB'], thickness)

        cv2.line(frame, landmarks[positions[3]],landmarks[positions[4]],colours['THUMB'],thickness)
        cv2.circle(frame, landmarks[positions[4]], outer_radius+tips, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[4]], inner_radius+tips, colours['THUMB'], -1)


        #For Index
        cv2.line(frame, landmarks[positions[5]],landmarks[positions[6]],colours['INDEX'],thickness)
        cv2.circle(frame, landmarks[positions[6]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[6]], inner_radius, colours['INDEX'], thickness)

        cv2.line(frame, landmarks[positions[6]],landmarks[positions[7]],colours['INDEX'],thickness)
        cv2.circle(frame, landmarks[positions[7]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[7]], inner_radius, colours['INDEX'], thickness)

        cv2.line(frame, landmarks[positions[7]],landmarks[positions[8]],colours['INDEX'],thickness)
        cv2.circle(frame, landmarks[positions[8]], outer_radius+tips, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[8]], inner_radius+tips, colours['INDEX'], -1)


        # For Middle
        cv2.line(frame, landmarks[positions[9]], landmarks[positions[10]], colours['MIDDLE'], thickness)
        cv2.circle(frame, landmarks[positions[10]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[10]], inner_radius, colours['MIDDLE'], thickness)

        cv2.line(frame, landmarks[positions[10]], landmarks[positions[11]], colours['MIDDLE'], thickness)
        cv2.circle(frame, landmarks[positions[11]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[11]], inner_radius, colours['MIDDLE'], thickness)

        cv2.line(frame, landmarks[positions[11]], landmarks[positions[12]], colours['MIDDLE'], thickness)
        cv2.circle(frame, landmarks[positions[12]], outer_radius+tips, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[12]], inner_radius+tips, colours['MIDDLE'], -1)


            #For Ring
        cv2.line(frame, landmarks[positions[13]], landmarks[positions[14]], colours['RING'], thickness)
        cv2.circle(frame, landmarks[positions[14]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[14]], inner_radius, colours['RING'], thickness)

        cv2.line(frame, landmarks[positions[14]], landmarks[positions[15]], colours['RING'], thickness)
        cv2.circle(frame, landmarks[positions[15]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[15]], inner_radius, colours['RING'], thickness)

        cv2.line(frame, landmarks[positions[15]], landmarks[positions[16]], colours['RING'], thickness)
        cv2.circle(frame, landmarks[positions[16]], outer_radius+tips, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[16]], inner_radius+tips, colours['RING'], -1)


        #For Pinky
        cv2.line(frame, landmarks[positions[17]], landmarks[positions[18]], colours['PINKY'], thickness)
        cv2.circle(frame, landmarks[positions[18]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[18]], inner_radius, colours['PINKY'], thickness)

        cv2.line(frame, landmarks[positions[18]], landmarks[positions[19]], colours['PINKY'], thickness)
        cv2.circle(frame, landmarks[positions[19]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[19]], inner_radius, colours['PINKY'], thickness)

        cv2.line(frame, landmarks[positions[19]], landmarks[positions[20]], colours['PINKY'], thickness)
        cv2.circle(frame, landmarks[positions[20]], outer_radius+tips, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[20]], inner_radius+tips, colours['PINKY'], -1)


        #For Palm
        cv2.line(frame, landmarks[positions[0]], landmarks[positions[1]], colours['PALM'], thickness)
        cv2.circle(frame, landmarks[positions[1]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[1]], inner_radius, colours['PALM'], thickness)

        cv2.line(frame, landmarks[positions[0]], landmarks[positions[5]], colours['PALM'], thickness)
        cv2.circle(frame, landmarks[positions[5]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[5]], inner_radius, colours['PALM'], thickness)

        cv2.line(frame, landmarks[positions[0]], landmarks[positions[17]], colours['PALM'], thickness)
        cv2.circle(frame, landmarks[positions[0]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[0]], inner_radius, colours['PALM'], thickness)

        cv2.line(frame, landmarks[positions[5]], landmarks[positions[9]], colours['PALM'], thickness)
        cv2.circle(frame, landmarks[positions[9]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[9]], inner_radius, colours['PALM'], thickness)

        cv2.line(frame, landmarks[positions[9]], landmarks[positions[13]], colours['PALM'], thickness)
        cv2.circle(frame, landmarks[positions[13]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[13]], inner_radius, colours['PALM'], thickness)

        cv2.line(frame, landmarks[positions[13]], landmarks[positions[17]], colours['PALM'], thickness)
        cv2.circle(frame, landmarks[positions[17]], outer_radius, (255, 255, 255), thickness)
        cv2.circle(frame, landmarks[positions[17]], inner_radius, colours['PALM'], thickness)

    return frame