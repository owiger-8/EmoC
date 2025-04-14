# EmoC: Emotion-Aware Conversational Agent

EmoC is a prototype system that combines **facial emotion recognition** with an **emotion-aware chatbot** to deliver contextually appropriate responses. It addresses modern challenges in human-computer interaction by enabling machines to adapt to users' emotional states, with applications in mental health support, customer service, and personalized AI companions.

---

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Workflow](#workflow)
- [Dataset Details](#dataset-details)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Key Features
- **Facial Emotion Recognition**: Trained on the CK+ dataset to detect 7 core emotions (anger, contempt, disgust, fear, happiness, sadness, surprise).
- **Fine-Tuned Chatbot**: Uses `newdata.json` to train TinyLLama for emotion-sensitive dialogue generation.
- **End-to-End Pipeline**: Integrates emotion detection and conversational AI in `main.py`.

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended)

### Steps
1. Clone repository:
  ```
  git clone https://github.com/yourusername/EmoC.git  
  cd EmoC
  ```

3. Install dependencies:
   pip install -r requirements.txt

4. Download assets:
- **CK+ Dataset**: Extract `ck_plus_dataset.zip` to `data/` directory
- **Pre-trained Models**:
  ```
  huggingface-cli download psxog/EmoC emotion_model.pth --local-dir models/
  ```

---

## Workflow
### 1. Emotion Recognition
- **Model**: CNN trained on CK+ dataset (`train_emotion_model.py`)
- **Input**: Real-time webcam feed or static images
- **Output**: Emotion labels with confidence scores

### 2. Response Generation
- **Model**: TinyLLama fine-tuned with `newdata.json` (`fine_tune_tinyllama_improved.py`)
- **Integration**: Emotion labels from Step 1 guide response selection

Simplified main.py workflow
emotion = emotion_model.detect(frame)
response = chatbot.generate_response(user_input, emotion)
print(f"EmoC: {response}")


---

## Dataset Details
| Dataset         | Purpose                          | Format       | Source                     |
|-----------------|----------------------------------|--------------|----------------------------|
| CK+             | Train emotion detection model    | 593 video clips | Public research dataset    |
| newdata.json    | Fine-tune chatbot responses      | JSON         | Custom conversation pairs  |

---

## Project Structure

EmoC/
├── data/ # CK+ dataset directory
├── models/ # emotion_model.pth
├── ck_plus_dataset.zip # Raw dataset
├── emotion_recognition.py # Inference script
├── fine_tune_tinyllama_improved.py
├── main.py # Entry point
├── newdata.json # Custom chatbot data
├── requirements.txt # Python dependencies
└── train_emotion_model.py # Training script


---

## Future Improvements
- Add real-time webcam integration
- Implement multi-modal emotion analysis (text + facial cues)
- Expand dataset with diverse demographic samples
- Optimize model for mobile deployment

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Acknowledgments
Special thanks to the creators of the CK+ dataset and the developers of TinyLLama for their contributions to this project.

---

# How to Use EmoC
1. **Run EmoC**:
   python main.py
2. **Interact**:
- EmoC will detect your emotions and respond accordingly.
- You can input text to engage in conversation.

---

# Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request describing your changes.

---

# References
- CK+ Dataset: [Extended Cohn-Kanade Dataset](https://www.pitt.edu/~emotion/ck-spread.htm)
- TinyLLama: [Hugging Face Model Hub](https://huggingface.co/models)


   

