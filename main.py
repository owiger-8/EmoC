import os
import json
import torch
import cv2
import numpy as np
import threading
import time
import queue
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext, font, messagebox
from PIL import Image, ImageTk
from transformers import AutoModelForCausalLM, AutoTokenizer
from emotion_recognition import EmotionRecognitionModel
import pyttsx3  # Offline TTS
import speech_recognition as sr

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SRRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def listen(self):
        with self.microphone as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

        try:
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
        except sr.UnknownValueError:
            print("Speech was unintelligible.")
        except sr.RequestError as e:
            print(f"Recognition request failed: {e}")

        return None

class EmoC:
    def __init__(self, root):
        self.root = root
        self.root.title("Emo C")
        self.root.configure(bg="black")
        self.root.geometry("800x600")

        self.setup_ui()
        self.emotion_model = EmotionRecognitionModel()
        self.emotion_model.to(device)

        self.model_loaded = False
        if os.path.exists("./fine_tuned_tinyllama"):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_tinyllama")
                self.llm = AutoModelForCausalLM.from_pretrained(
                    "./fine_tuned_tinyllama",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self.llm.to(device)
                self.model_loaded = True
                print("Loaded fine-tuned TinyLlama model")
            except Exception as e:
                print(f"Error loading fine-tuned model: {e}")
                print("Using placeholder responses instead")
        else:
            print("Fine-tuned model not found. Using placeholder responses instead")

        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self.tts_loop)
        self.tts_thread.daemon = True
        self.tts_thread.start()

        self.speech_recognizer = SRRecognizer()
        self.message_queue = queue.Queue()

        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        self.current_emotion = "neutral"

        self.camera_active = True
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()

        self.speech_active = True
        self.is_listening = False
        self.speech_thread = None

        self.processing_thread = threading.Thread(target=self.process_messages)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.root.after(100, self.check_messages)
        self.root.after(33, self.update_webcam_display)

        self.setup_text_input()

    def setup_ui(self):
        self.main_container = tk.Frame(self.root, bg="black")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.left_column = tk.Frame(self.main_container, bg="black")
        self.left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20, 10), pady=20)

        self.right_column = tk.Frame(self.main_container, bg="black", width=240)
        self.right_column.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 20), pady=20)
        self.right_column.pack_propagate(False)

        self.webcam_label = tk.Label(self.right_column, bg="black")
        self.webcam_label.pack(fill=tk.BOTH, expand=True)

        self.emotion_label = tk.Label(self.right_column, text="Emotion: neutral", bg="black", fg="white", font=("Courier", 12))
        self.emotion_label.pack(pady=(10, 0))

        self.chat_display = scrolledtext.ScrolledText(self.left_column, bg="black", fg="white", font=("Courier", 12), wrap=tk.WORD, bd=0, highlightthickness=0)
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.configure(state=tk.DISABLED)

        self.chat_display.tag_configure("user", foreground="white")
        self.chat_display.tag_configure("bot", foreground="white")
        self.chat_display.tag_configure("system", foreground="yellow")

        self.add_message("Emo C", "Hello! I'm Emo C, your emotional companion. I can see how you're feeling and we can talk about it.")
        self.add_system_message("Using speech_recognition package for voice input. Click the microphone button to start listening.")

    def setup_text_input(self):
        self.input_frame = tk.Frame(self.left_column, bg="black")
        self.input_frame.pack(fill=tk.X, pady=(10, 0))

        self.text_input = tk.Entry(self.input_frame, bg="#333", fg="white", font=("Courier", 12), bd=1, insertbackground="white")
        self.text_input.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 10))
        self.text_input.bind("<Return>", self.on_text_submit)

        self.send_button = tk.Button(self.input_frame, text="Send", bg="#555", fg="white", font=("Courier", 12), bd=0, padx=10, command=self.on_text_submit)
        self.send_button.pack(side=tk.RIGHT)

        self.mic_button = tk.Button(self.input_frame, text="🎤", bg="#555", fg="white", font=("Courier", 12), bd=0, padx=10, command=self.toggle_listening)
        self.mic_button.pack(side=tk.RIGHT, padx=(0, 10))

        self.listening_label = tk.Label(self.left_column, text="Click microphone to start listening", bg="black", fg="white", font=("Courier", 10))
        self.listening_label.pack(pady=(5, 0))

    def toggle_listening(self):
        if self.is_listening:
            return
        self.is_listening = True
        self.listening_label.config(text="Listening...", fg="green")
        self.mic_button.config(bg="#080")
        self.speech_thread = threading.Thread(target=self.listen_for_speech)
        self.speech_thread.daemon = True
        self.speech_thread.start()

    def listen_for_speech(self):
        try:
            recognized_text = self.speech_recognizer.listen()
            if recognized_text:
                self.add_system_message("Recognized speech: " + recognized_text)
                self.message_queue.put(("user_speech", recognized_text))
            else:
                self.add_system_message("Could not understand speech.")
        except Exception as e:
            self.add_system_message(f"Error in speech recognition: {str(e)}")
        finally:
            self.is_listening = False
            self.root.after(0, lambda: self.listening_label.config(text="Click microphone to start listening", fg="white"))
            self.root.after(0, lambda: self.mic_button.config(bg="#555"))

    def on_text_submit(self, event=None):
        user_input = self.text_input.get().strip()
        if user_input:
            self.text_input.delete(0, tk.END)
            self.message_queue.put(("user_speech", user_input))

    def add_message(self, sender, message):
        self.chat_display.configure(state=tk.NORMAL)
        if sender == "user":
            self.chat_display.insert(tk.END, f"you: {message}\n\n", "user")
        else:
            self.chat_display.insert(tk.END, f"Emo C: {message}\n\n", "bot")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)

    def add_system_message(self, message):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"System: {message}\n\n", "system")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)

    def camera_loop(self):
        time.sleep(1)
        last_emotion_time = time.time()
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                continue
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if time.time() - last_emotion_time > 2:
                emotion = self.emotion_model.predict(frame)
                if emotion != self.current_emotion:
                    self.current_emotion = emotion
                    print(f"Detected emotion: {self.current_emotion}")
                    self.message_queue.put(("emotion", emotion))
                    self.root.after(0, lambda: self.emotion_label.config(text=f"Emotion: {self.current_emotion}"))
                last_emotion_time = time.time()
            time.sleep(0.03)

    def update_webcam_display(self):
        if self.current_frame is not None:
            small_frame = cv2.resize(self.current_frame, (240, 180))
            img = Image.fromarray(small_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)
        self.root.after(33, self.update_webcam_display)

    def process_messages(self):
        while True:
            try:
                msg_type, content = self.message_queue.get(timeout=0.5)
                if msg_type == "emotion":
                    pass
                elif msg_type == "user_speech":
                    user_input = content
                    self.root.after(0, lambda: self.add_message("user", user_input))
                    response = self.generate_response(user_input, self.current_emotion)
                    self.root.after(0, lambda: self.add_message("Emo C", response))
                    self.speak_offline(response)
                self.message_queue.task_done()
            except queue.Empty:
                pass

    def generate_response(self, user_input, emotion):
        if emotion is None:
            emotion = "neutral"
        if not self.model_loaded:
            import random
            responses = {
                "anger": ["I can see you're feeling angry. Would you like to talk about what's bothering you?"],
                "disgust": ["I sense you're feeling disgusted. What's causing this reaction?"],
                "fear": ["I can see you might be feeling afraid. Remember that you're safe right now."],
                "happiness": ["You seem happy! That's wonderful to see."],
                "sadness": ["I notice you seem a bit down. Would you like to talk about it?"],
                "surprise": ["You look surprised! Did something unexpected happen?"],
                "neutral": ["How are you feeling today?"]
            }
            return random.choice(responses.get(emotion, responses["neutral"]))
        prompt = f"input: {user_input}\nemotion: {emotion}\noutput:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = self.llm.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        full_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = full_output.split("output:")[-1].strip()
        return response

    def speak_offline(self, text):
        self.tts_queue.put(text)

    def tts_loop(self):
        while True:
            try:
                text = self.tts_queue.get()
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                self.tts_queue.task_done()
            except Exception as e:
                print(f"TTS Error: {e}")

    def check_messages(self):
        self.root.after(100, self.check_messages)

    def on_closing(self):
        self.camera_active = False
        self.speech_active = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmoC(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
