"""
AI Mood DJ — Hand → Face → Music
Without MediaPipe (uses only OpenCV)
Works on macOS (M1/M2)

Requirements:
pip install opencv-python pygame
"""

import cv2
from pygame import mixer
import tkinter as tk
from threading import Thread
import time
import os

# ---------- SONGS FOR EACH MOOD ----------
MOOD_SONGS = {
    "happy": "songs/happy.mp3",
    "neutral": "songs/lofi.mp3",
    "sad": "songs/sad.mp3"
}

# ---------- INITIALIZE MIXER ----------
mixer.init()

def play_song(mood):
    song = MOOD_SONGS.get(mood, "songs/lofi.mp3")
    if not os.path.exists(song):
        print(f"[WARN] Missing file: {song}")
        return
    try:
        mixer.music.load(song)
        mixer.music.play()
        print(f"[INFO] Playing {mood} song → {song}")
    except Exception as e:
        print(f"[ERROR] Could not play {song}: {e}")

# ---------- PHASE 2: FACE & MOOD DETECTION ----------
def start_face_detection():
    # Load OpenCV's pre-trained Haar cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting mood detection using OpenCV...")

    mood_detected = False
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera not detected.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = gray[y:y + h, x:x + w]

            smiles = smile_cascade.detectMultiScale(face_roi, 1.8, 20)

            if len(smiles) > 0:
                mood = "happy"
            else:
                # Wait a few seconds before deciding neutral/sad
                if time.time() - start_time > 5:
                    mood = "sad"
                else:
                    mood = "neutral"

            cv2.putText(frame, f"Mood: {mood}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            play_song(mood)
            mood_detected = True
            break

        cv2.imshow("AI Mood DJ - Mood Detection", frame)

        if mood_detected:
            cv2.waitKey(5000)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- PHASE 1: HAND DETECTION (Basic Motion Detection) ----------
def start_hand_detection():
    print("[INFO] Show your hand in front of camera to continue...")
    cap = cv2.VideoCapture(0)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
    motion_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera not detected.")
            break

        frame = cv2.flip(frame, 1)
        fg_mask = bg_subtractor.apply(frame)

        # Count nonzero pixels (movement)
        motion = cv2.countNonZero(fg_mask)
        h, w = frame.shape[:2]
        box_area = (h * w) // 15

        cv2.putText(frame, "Show your hand to start", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("AI Mood DJ - Hand Detection", frame)

        if motion > box_area:
            motion_detected = True
            print("[INFO] Hand/Motion detected ✅")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if motion_detected:
        start_face_detection()

# ---------- GUI ----------
def start_gui():
    root = tk.Tk()
    root.title("AI Mood DJ")
    root.geometry("600x400")
    root.configure(bg="#111")

    tk.Label(root, text="🎧 AI Mood DJ", bg="#111", fg="#00ffcc",
             font=("Poppins", 26, "bold")).pack(pady=40)
    tk.Label(root, text="Wave your hand & smile — AI will play your vibe!",
             bg="#111", fg="#ccc", font=("Poppins", 12)).pack(pady=10)

    def proceed():
        root.destroy()
        Thread(target=start_hand_detection).start()

    tk.Button(root, text="Proceed", bg="#00bcd4", fg="white",
              font=("Poppins", 14, "bold"), relief="flat",
              padx=20, pady=10, command=proceed).pack(pady=50)

    root.mainloop()

# ---------- RUN ----------
if __name__ == "__main__":
    start_gui()
