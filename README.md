# Virtual mouse 

This is a first stage project functional python backend code for a virtual mouse. Using this you can control your mouse with your hand movements, finger-tip movements, and voice commands.

## Libraries Used

| Library | Purpose |
|---------|---------|
| `opencv-python` | Capturing and processing webcam frames |
| `mediapipe` | Hand landmark detection |
| `pyautogui` | Controlling mouse cursor and clicks |
| `SpeechRecognition` | **Voice recognition** – listens to microphone input and converts speech to text so the app can respond to spoken commands |
| `PyAudio` | Microphone audio input (required by `SpeechRecognition`) |

> **Voice Recognition library:** `SpeechRecognition` (imported as `import speech_recognition as sr`).  
> It uses Google's Speech Recognition API under the hood and works with any microphone accessible via `PyAudio`.

## Supported Voice Commands

| Say | Action |
|-----|--------|
| "scroll up" | Scroll the page upward |
| "scroll down" | Scroll the page downward |
| "quit" / "exit" | Close the application |

## How to use this

1. Clone the repo
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Enjoy – control the mouse with your hand and issue voice commands!
