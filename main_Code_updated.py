import cv2
import numpy as np
import face_recognition
import os
import pyttsx3
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
from pynput.mouse import Controller, Button
import HandTrackingModule as htm
import time

face_folder = 'faces'
os.makedirs(face_folder, exist_ok=True)
attendance_file = 'Attendance.csv'

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def encode_faces():
    images, classNames = [], []
    for cl in os.listdir(face_folder):
        img = cv2.imread(f'{face_folder}/{cl}')
        if img is not None:
            images.append(img)
            classNames.append(os.path.splitext(cl)[0])

    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            continue
    return encodeList, classNames

def markAttendance(name):
    with open(attendance_file, 'a+') as f:
        f.seek(0)
        lines = f.readlines()
        nameList = [line.split(',')[0] for line in lines]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def register_new_face():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    speak("Please look at the camera.")
    time.sleep(2)
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face_crop = frame[top:bottom, left:right]
            name = simpledialog.askstring("Register", "Enter your name:")
            if name:
                cv2.imwrite(f'{face_folder}/{name}.jpg', face_crop)
                speak("Registration successful.")
            else:
                speak("Registration cancelled.")
        else:
            speak("No face detected.")
    else:
        speak("Camera failed.")
    cap.release()
    cv2.destroyAllWindows()

def face_recognition_validation():
    encodeListKnown, classNames = encode_faces()
    if not encodeListKnown:
        speak("No registered faces.")
        messagebox.showerror("Error", "No faces registered.")
        return False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    speak("Starting face recognition.")

    while True:
        success, img = cap.read()
        if not success:
            continue

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace in encodeCurFrame:
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            if len(faceDis) == 0:
                continue
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex]
                markAttendance(name)
                speak(f"Welcome {name}")
                cap.release()
                cv2.destroyAllWindows()
                return True

        cv2.imshow('Face Recognition', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    speak("Face not recognized.")
    return False

def start_virtual_mouse():
    wCam, hCam = 640, 480
    frameR = 100
    smoothening = 7
    pTime, plocX, plocY = 0, 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = htm.handDetector(maxHands=1)
    mouse = Controller()
    screen_width, screen_height = 1920, 1080

    # New variables for improved gesture control
    click_time = 0
    scroll_time = 0
    drag_active = False

    speak("Virtual Mouse Activated")

    while True:
        success, img = cap.read()
        if not success:
            continue

        img = detector.findHands(img)
        lmList, _ = detector.findPosition(img)

        if lmList:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            x0, y0 = lmList[4][1:]
            fingers = detector.fingersUp()
            current_time = time.time()
            action_taken = False

            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

            # Move Cursor
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (screen_width, 0))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, screen_height))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                mouse.position = (clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY
                action_taken = True

            # Left Click
            if not action_taken and fingers[1] == 1 and fingers[2] == 1:
                length, img, _ = detector.findDistance(8, 12, img)
                if length < 40 and (current_time - click_time) > 0.5:
                    mouse.click(Button.left, 1)
                    click_time = current_time
                    action_taken = True

            # Right Click
            if not action_taken and fingers == [0, 0, 0, 0, 1] and (current_time - click_time) > 0.5:
                mouse.click(Button.right, 1)
                click_time = current_time
                action_taken = True

            # Click and Drag
            if not action_taken and fingers[0] == 1 and fingers[1] == 1:
                if not drag_active:
                    mouse.press(Button.left)
                    drag_active = True
                x3 = np.interp(x1, (frameR, wCam - frameR), (screen_width, 0))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, screen_height))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                mouse.position = (clocX, clocY)
                plocX, plocY = clocX, clocY
                action_taken = True
            else:
                if drag_active:
                    mouse.release(Button.left)
                    drag_active = False

            # Scroll
            if not action_taken and all(f == 1 for f in fingers):
                if y1 < hCam // 3:
                    mouse.scroll(0, 2)
                elif y1 > (2 * hCam) // 3:
                    mouse.scroll(0, -2)
                scroll_time = current_time
                time.sleep(0.3)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime != 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Virtual Mouse", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def gui_window():
    root = tk.Tk()
    root.title("AI Virtual Mouse")
    root.geometry("400x250")

    tk.Label(root, text="AI Virtual Mouse", font=("Arial", 16)).pack(pady=20)

    def login():
        root.destroy()
        result = face_recognition_validation()
        if result:
            start_virtual_mouse()

    def register():
        register_new_face()

    tk.Button(root, text="Login", command=login, width=20, height=2).pack(pady=10)
    tk.Button(root, text="Register New Face", command=register, width=20, height=2).pack(pady=10)
    tk.Button(root, text="Exit", command=root.quit, width=20, height=2).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    gui_window()
