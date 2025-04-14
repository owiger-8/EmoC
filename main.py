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
from tkinter import scrolledtext, font
from PIL import Image, ImageTk
from transformers import AutoModelForCausalLM, AutoTokenizer
from emotion_recognition import EmotionRecognitionModel
import pyttsx3  # Offline TTS
import pyaudio  # For audio input
import wave
import tempfile
from vosk import Model, KaldiRecognizer  # Offline speech recognition

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EmoC:
    def __init__(self, root):
        self.root = root
        self.root.title("Emo C")
        self.root.configure(bg="black")
        self.root.geometry("800x600")
        
        # Set up the UI
        self.setup_ui()
        
        # Initialize the emotion recognition model
        self.emotion_model = EmotionRecognitionModel()
        self.emotion_model.to(device)
        
        # Check if fine-tuned model exists, otherwise use a placeholder response generator
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
        
        # Initialize offline text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        
        # Initialize offline speech recognition
        self.setup_speech_recognition()
        
        # Message queue for communication between threads
        self.message_queue = queue.Queue()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        self.current_emotion = "neutral"
        
        # Start the camera thread
        self.camera_active = True
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        # Start the speech recognition thread
        self.speech_active = True
        self.speech_thread = threading.Thread(target=self.speech_recognition_loop)
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
        # Start the message processing thread
        self.processing_thread = threading.Thread(target=self.process_messages)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Periodically check for new messages and update webcam display
        self.root.after(100, self.check_messages)
        self.root.after(33, self.update_webcam_display)  # ~30 fps
        
        # Setup text input
        self.setup_text_input()
    
    def setup_ui(self):
        # Main container with two columns
        self.main_container = tk.Frame(self.root, bg="black")
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left column for chat (70% width)
        self.left_column = tk.Frame(self.main_container, bg="black")
        self.left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20, 10), pady=20)
        
        # Right column for webcam (30% width)
        self.right_column = tk.Frame(self.main_container, bg="black", width=240)
        self.right_column.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 20), pady=20)
        self.right_column.pack_propagate(False)  # Prevent the frame from shrinking
        
        # Create a label for the webcam display
        self.webcam_label = tk.Label(self.right_column, bg="black")
        self.webcam_label.pack(fill=tk.BOTH, expand=True)
        
        # Create a label for the current emotion
        self.emotion_label = tk.Label(
            self.right_column, 
            text="Emotion: neutral",
            bg="black",
            fg="white",
            font=("Courier", 12)
        )
        self.emotion_label.pack(pady=(10, 0))
        
        # Create a scrolled text widget for the chat
        self.chat_display = scrolledtext.ScrolledText(
            self.left_column,
            bg="black",
            fg="white",
            font=("Courier", 12),
            wrap=tk.WORD,
            bd=0,
            highlightthickness=0
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.configure(state=tk.DISABLED)
        
        # Define tags for different speakers
        self.chat_display.tag_configure("user", foreground="white")
        self.chat_display.tag_configure("bot", foreground="white")
        
        # Welcome message
        self.add_message("Emo C", "Hello! I'm Emo C, your emotional companion. I can see how you're feeling and we can talk about it.")
    
    def setup_text_input(self):
        # Create a frame for the text input
        self.input_frame = tk.Frame(self.left_column, bg="black")
        self.input_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create a text entry widget
        self.text_input = tk.Entry(
            self.input_frame,
            bg="#333",
            fg="white",
            font=("Courier", 12),
            bd=1,
            insertbackground="white"  # cursor color
        )
        self.text_input.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 10))
        self.text_input.bind("<Return>", self.on_text_submit)
        
        # Create a send button
        self.send_button = tk.Button(
            self.input_frame,
            text="Send",
            bg="#555",
            fg="white",
            font=("Courier", 12),
            bd=0,
            padx=10,
            command=self.on_text_submit
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Create a microphone button for speech input
        self.mic_button = tk.Button(
            self.input_frame,
            text="🎤",
            bg="#555",
            fg="white",
            font=("Courier", 12),
            bd=0,
            padx=10,
            command=self.toggle_listening
        )
        self.mic_button.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Listening status indicator
        self.listening_label = tk.Label(
            self.left_column,
            text="Speech recognition active",
            bg="black",
            fg="green",
            font=("Courier", 10)
        )
        self.listening_label.pack(pady=(5, 0))
    
    def setup_speech_recognition(self):
        # Check if Vosk model exists, if not, inform the user
        model_path = "vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path):
            print(f"Vosk model not found at {model_path}. Please download it from https://alphacephei.com/vosk/models")
            print("For now, using a placeholder for speech recognition")
            self.vosk_model = None
        else:
            try:
                self.vosk_model = Model(model_path)
                print(f"Loaded Vosk model from {model_path}")
            except Exception as e:
                print(f"Error loading Vosk model: {e}")
                self.vosk_model = None
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_listening = True
    
    def toggle_listening(self):
        self.is_listening = not self.is_listening
        if self.is_listening:
            self.listening_label.config(text="Speech recognition active", fg="green")
            self.mic_button.config(bg="#555")
        else:
            self.listening_label.config(text="Speech recognition paused", fg="red")
            self.mic_button.config(bg="#800")
    
    def on_text_submit(self, event=None):
        user_input = self.text_input.get().strip()
        if user_input:
            self.text_input.delete(0, tk.END)
            self.message_queue.put(("user_speech", user_input))
    
    def add_message(self, sender, message):
        self.chat_display.configure(state=tk.NORMAL)
        
        # Add timestamp and sender
        if sender == "user":
            self.chat_display.insert(tk.END, f"you: {message}\n\n", "user")
        else:
            self.chat_display.insert(tk.END, f"Emo C: {message}\n\n", "bot")
        
        # Auto-scroll to the bottom
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)
    
    def camera_loop(self):
        # Wait for camera to initialize
        time.sleep(1)
        
        last_emotion_time = time.time()
        
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                time.sleep(1)
                continue
            
            # Store the current frame for display
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame every 2 seconds to detect emotion
            if time.time() - last_emotion_time > 2:
                emotion = self.emotion_model.predict(frame)
                
                # Only update if emotion changed
                if emotion != self.current_emotion:
                    self.current_emotion = emotion
                    print(f"Detected emotion: {self.current_emotion}")
                    self.message_queue.put(("emotion", emotion))
                    
                    # Update emotion label
                    self.root.after(0, lambda: self.emotion_label.config(text=f"Emotion: {self.current_emotion}"))
                
                last_emotion_time = time.time()
            
            time.sleep(0.03)  # ~30 fps
    
    def speech_recognition_loop(self):
        if self.vosk_model is None:
            # Placeholder for when Vosk model is not available
            while self.speech_active:
                time.sleep(1)
            return
        
        # Create recognizer
        rec = KaldiRecognizer(self.vosk_model, 16000)
        
        # Open microphone stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8000
        )
        self.stream.start_stream()
        
        while self.speech_active:
            if not self.is_listening:
                time.sleep(0.1)
                continue
                
            try:
                data = self.stream.read(4000, exception_on_overflow=False)
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result.get("text", "").strip():
                        recognized_text = result["text"]
                        print(f"Recognized: {recognized_text}")
                        self.message_queue.put(("user_speech", recognized_text))
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                time.sleep(0.1)
        
        # Clean up
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
    
    def update_webcam_display(self):
        if self.current_frame is not None:
            # Resize the frame to fit the webcam display
            small_frame = cv2.resize(self.current_frame, (240, 180))
            
            # Convert to PhotoImage
            img = Image.fromarray(small_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the label
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)
        
        # Schedule the next update
        self.root.after(33, self.update_webcam_display)
    
    def process_messages(self):
        while True:
            try:
                msg_type, content = self.message_queue.get(timeout=0.5)
                
                if msg_type == "emotion":
                    # We don't need to respond to every emotion change
                    pass
                
                elif msg_type == "user_speech":
                    user_input = content
                    self.root.after(0, lambda: self.add_message("user", user_input))
                    
                    # Generate response using the LLM or placeholder
                    response = self.generate_response(user_input, self.current_emotion)
                    
                    # Add the response to the UI
                    self.root.after(0, lambda: self.add_message("Emo C", response))
                    
                    # Speak the response using offline TTS
                    self.speak_offline(response)
                
                self.message_queue.task_done()
            
            except queue.Empty:
                pass
    
    def generate_response(self, user_input, emotion):
        if emotion is None:
            emotion = "neutral"
        
        # If model is not loaded, use placeholder responses
        if not self.model_loaded:
            # Simple rule-based responses based on emotion
            responses = {
                "anger": [
                    "I can see you're feeling angry. Would you like to talk about what's bothering you?",
                    "It's okay to feel angry sometimes. I'm here to listen if you want to talk about it.",
                    "I notice you seem upset. Is there something specific that triggered this feeling?"
                ],
                "disgust": [
                    "I sense you're feeling disgusted. What's causing this reaction?",
                    "Something seems to be bothering you. Would you like to share what it is?",
                    "I'm here to listen if you want to talk about what's making you uncomfortable."
                ],
                "fear": [
                    "I can see you might be feeling afraid. Remember that you're safe right now.",
                    "It's okay to feel scared sometimes. Would you like to talk about what's worrying you?",
                    "I'm here with you. Would it help to talk about what's causing your concern?"
                ],
                "happiness": [
                    "You seem happy! That's wonderful to see.",
                    "Your smile is contagious! What's bringing you joy today?",
                    "I'm glad to see you're in good spirits. Would you like to share what's making you happy?"
                ],
                "sadness": [
                    "I notice you seem a bit down. Would you like to talk about it?",
                    "It's okay to feel sad sometimes. I'm here to listen if you need someone to talk to.",
                    "I'm here for you. Would sharing what's on your mind help you feel better?"
                ],
                "surprise": [
                    "You look surprised! Did something unexpected happen?",
                    "I notice you seem taken aback. Would you like to share what surprised you?",
                    "Something seems to have caught you off guard. I'm here if you want to talk about it."
                ],
                "neutral": [
                    "How are you feeling today?",
                    "Is there anything specific you'd like to talk about?",
                    "I'm here to chat whenever you're ready."
                ]
            }
            
            # Choose a random response based on the detected emotion
            import random
            return random.choice(responses.get(emotion, responses["neutral"]))
        
        # Format the input for the model
        prompt = f"input: {user_input}\nemotion: {emotion}\noutput:"
        print(prompt )
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = self.llm.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # Decode the response
        full_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the output part
        response = full_output.split("output:")[-1].strip()
        print(response)
        return response
    
    def speak_offline(self, text):
        # Use pyttsx3 for offline text-to-speech
        def speak_thread():
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        
        # Run TTS in a separate thread to avoid blocking the UI
        tts_thread = threading.Thread(target=speak_thread)
        tts_thread.daemon = True
        tts_thread.start()
    
    def check_messages(self):
        # This function runs in the main thread and checks for UI updates
        self.root.after(100, self.check_messages)
    
    def on_closing(self):
        self.camera_active = False
        self.speech_active = False
        
        # Clean up resources
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'audio'):
            self.audio.terminate()
        
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmoC(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
