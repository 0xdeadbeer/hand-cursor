#!/usr/bin/python3 
import os 
import sys 
import math 
import random 
import cv2
import mediapipe as mp
import pyautogui

def main(): 
    camera_cap = cv2.VideoCapture(0)    
    screen_size = pyautogui.size()

    hands_solution = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing = mp.solutions.drawing_utils
    hands = hands_solution.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while camera_cap.isOpened(): 
        success, image = camera_cap.read() 
        if not success:
            print("Ignoring empty camera frame..")
            continue

        image = cv2.flip(image, 1)

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if (results.multi_hand_landmarks):
            for hand_landmarks in results.multi_hand_landmarks:

                cursor_hand = hand_landmarks.landmark[9]
                print(f"X -> {cursor_hand.x} Y -> {cursor_hand.y}")
                cursor_position = [screen_size.width * cursor_hand.x, screen_size.height * cursor_hand.y]
                print(f"Cursor position: {cursor_position}") 
                pyautogui.moveTo(cursor_position[0], cursor_position[1])
                mp_drawing.draw_landmarks(image, 
                                            hand_landmarks, 
                                            hands_solution.HAND_CONNECTIONS, 
                                            mp_drawing_styles.get_default_hand_landmarks_style(),
                                            mp_drawing_styles.get_default_hand_connections_style())

        # cv2.imwrite("./hand-output.jpg", cv2.flip(annotated_image, 1))
        cv2.imshow("Media Hands", image)

        if (cv2.waitKey(5) & 0xFF == 27):
            break

    camera_cap.release()
    
if __name__ == "__main__":
    main() 
