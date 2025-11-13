"""
AI Mood DJ  —  Hand → Face → Music

Requirements:
pip install opencv-python mediapipe deepface pygame
"""

import cv2, mediapipe as mp
from deepface import DeepFace
from pygame import mixer
import tkinter as tk
from threading import Thread

# ---------- SONGS FOR EACH MOOD ----------
MOOD_SONGS = {
    "happy": "songs/happy.mp3",
    "sad": "songs/sad.mp3",
    "angry": "songs/chill.mp3",
    "surprise": "songs/excited.mp3",
    "neutral": "songs/lofi.mp3"
}

# ---------- PLAY SONG ----------
def play_song(emotion):
    mixer.init()
    song = MOOD_SONGS.get(emotion, "songs/lofi.mp3")
    mixer.music.load(song)
    mixer.music.play()

# ---------- PHASE 2: MOOD DETECTION ----------
def start_mood_detection():
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting mood detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.putText(frame, f"Mood: {emotion}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            play_song(emotion)
            cv2.imshow("AI Mood DJ - Mood Detection", frame)
            cv2.waitKey(5000)  # play a few seconds
            break  # detect once then stop
        except Exception as e:
            print("Waiting for face...")
            cv2.imshow("AI Mood DJ - Mood Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- PHASE 1: HAND DETECTION ----------
def start_hand_detection():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    print("[INFO] Show your hand inside the box to continue...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        h, w, _ = frame.shape
        # Draw the guide box
        x1, y1, x2, y2 = w//2 - 100, h//2 - 100, w//2 + 100, h//2 + 100
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Place your hand inside box", (x1 - 50, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                x_coords = [lm.x * w for lm in handLms.landmark]
                y_coords = [lm.y * h for lm in handLms.landmark]
                if all(x1 < x < x2 for x in x_coords) and all(y1 < y < y2 for y in y_coords):
                    cv2.putText(frame, "Hand Detected ✅", (x1, y2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cap.release()
                    cv2.destroyAllWindows()
                    start_mood_detection()
                    return

        cv2.imshow("AI Mood DJ - Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- GUI ----------
def start_gui():
    root = tk.Tk()
    root.title("AI Mood DJ")
    root.geometry("600x400")
    root.configure(bg="#111")

    tk.Label(root, text="🎧 AI Mood DJ", bg="#111", fg="#00ffcc",
             font=("Poppins", 26, "bold")).pack(pady=40)
    tk.Label(root, text="Let AI detect your mood and play the perfect song!",
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

