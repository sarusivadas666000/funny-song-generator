
<img width="3188" height="1202" alt="frame (3)" src="https://github.com/user-attachments/assets/517ad8e9-ad22-457d-9538-a9e62d137cd7" />


# Flowrica 🎯


## An application that detects flowers and generates songs inspired by the ones it recognizes.
### Team Name: Pythons


### Team Members
- Member 1: Saru S - ICCS College of Engineering ad Management
- Member 3: Stiny N S - ICCS College of Engineering and Management

### Project Description
A flower detecting application where it genereate  songs related to the detected flower. The song will be based on the flower or the flower name or its features 
### The Problem (that doesn't exist)
Solving the boredom of sitting alone.

### The Solution (that nobody asked for)
While generating the song by analysing the flower the user can hear to the song giving a piece of enjoyment. 
## Technical Details
### Technologies/Components Used
For Software:
- Language -Python,
- Framework- OpenCV, TensorFlow
- Libraries used- OS,time,threading,numpy,CV2,Tflite_runtime.interpreter.
- Tools used- VS Code, python interpreter,pip, pyinstaller,venv.

For Hardware:
- Web camera,Audio output device,computer.
- List specifications- Webcam: Minimum VGA resolution, USB connectivity,Audio device: Standard  built-in speakers/headphones,Computer: Should support running Python and OpenCV
- List tools required- Built-in Webcam,Headhones/Speakers for audio output, OS .
- 

### Implementation
For Software:
# Installation
commands- pip install pyinstaller
        -pip install numpy opencv-python tensorflow playsound
        -pip install numpy opencv-python tflite-runtime playsoun

# Run
commands - python your_main_script.py'  

### Project Documentation
For Software:

# Screenshots (Add at least 3)
<img width="1920" height="1080" alt="Screenshot from 2025-08-09 03-10-41" src="https://github.com/user-attachments/assets/aac1f91f-3b6d-4ac1-9057-e2a8e916bcea" />

This image shows a real-time flower recognition system in action,this model to identify flowers through a connected camera. In the preview window, it has successfully detected a dandelion displayed on a phone screen  the system requires 30 consecutive high-confidence detections (REQUIRED_COUNT = 30) before confirming an identification. Once confirmed, it stops scanning and plays a corresponding audio file.

 <img width="831" height="977" alt="Screenshot from 2025-08-09 03-12-23" src="https://github.com/user-attachments/assets/0c458506-c465-4b87-a294-fdb3f45d9512" />

Model loading logic that first attempts to use the lightweight tflite_runtime interpreter, falling back to the standard tensorflow.lite interpreter if needed.
Audio playback via the playsound library, ensuring the MP3 file plays to completion before continuing.
Helper functions for reading label files and initializing TensorFlow Lite models.
In the terminal, the system detects a dandelion with 100% confidence, maintains this recognition for 30 consecutive frames, and then triggers playback of assets/audio/dandelions.mp3.

<img width="831" height="993" alt="Screenshot from 2025-08-09 03-14-16" src="https://github.com/user-attachments/assets/77909249-48be-49f4-acc8-0b93699c123e" />

It is the datasets of the flower dandelions

# Diagrams
Workflow-<img width="1024" height="1536" alt="ChatGPT Image Aug 9, 2025, 03_59_28 AM" src="https://github.com/user-attachments/assets/03630482-0f39-4f67-b97e-8c491a6ba02b" />


This application captures live  frames using OpenCV and preprocesses them for TensorFlow input. A trained TensorFlow model detects flowers in each frame and outputs their location and confidence score. The decision logic verifies if any flower is confidently detected; if yes, it triggers song playback via a local audio file or text-to-speech engine. Meanwhile, OpenCV overlays bounding boxes and labels on the video feed to provide visual feedback. The video with overlays is displayed continuously until the user exits the application. This workflow ensures real-time flower detection paired with an interactive audio response.

For Hardware:

# Schematic & Circuit

Schematic -
<img width="1536" height="1024" alt="ChatGPT Image Aug 9, 2025, 05_11_46 AM" src="https://github.com/user-attachments/assets/ffeba33b-04cc-4fd1-86e9-cb87e7fbddb4" />

The schematic illustrates the detailed wiring of the hardware components including the camera interface, power regulation, and audio output circuit. It ensures proper voltage levels and signal integrity between components. Any control buttons or LEDs for user interaction are also represented here.




### Project Demo

# Video
https://github.com/user-attachments/assets/0fd60226-2d00-403e-98b1-7d0d8ac28a5b

This is an application built using OpenCV and TensorFlow that detects flowers and generates funny songs related to the detected flowers. When flowers like roses, dandelions, or sunflowers are recognized, the app plays humorous songs inspired by each flower. The demo showcases the flower detection in action and the corresponding funny tunes that bring a playful twist to the experience.





## Team Contributions
- Saru S: Flower Detection & Image Processing
- Stiny N S : Song Generation & Application Integration


---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--25-25?link=https%3A%2F%2Fwww.tinkerhub.org%2Fevents%2FQ2Q1TQKX6Q%2FUseless%2520Projects)


