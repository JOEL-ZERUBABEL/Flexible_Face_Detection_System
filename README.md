🚀 Project Overview

This project is an AI-powered Face Detection and Verification System that combines Deep Learning, Mediapipe FaceMesh, Liveness Detection, and Face Recognition into a single unified pipeline.

It supports:

✔ Live Webcam Face Detection
✔ Photo Upload Based Face Verification
✔ DeepFace-powered Face Recognition (ArcFace, RetinaFace)
✔ Eye-Blink Based Liveness Detection (EAR)
✔ FaceMesh Landmark Extraction
✔ Feature Embedding Comparison (Cosine Similarity / Euclidean Distance)
✔ Multiple ML & CV Models at Once (Hybrid System)

This project is designed to demonstrate strong industry-level AI engineering skills, including computer vision, machine learning, embeddings, and real-time systems.

✨ Key Features

🔍 1. Real-Time Face Detection

Uses Mediapipe FaceMesh and CVZone FaceDetector
Extracts 468 facial landmarks
Detects eyes, irises, lips, jaw structure, and contours in real time

🧠 2. DeepFace-Based Face Verification

ArcFace embeddings
RetinaFace detection backend
Cosine similarity for identity verification

Works for:

Image-to-Image verification
Webcam-to-Image verification

👁️ 3. Eye Blink Liveness Detection

Uses Eye Aspect Ratio (EAR)

Prevents spoofing using photos or printed images

Detects:

Blinking
Eye openness score
Potential spoof attacks

🎯 4. Embedding-Based Face Comparison

Cosine similarity
Euclidean distance
ArcFace 512-dimensional embeddings
Supports threshold-based authentication

📸 5. Dual Input Modes

Upload Image Mode → Verify uploaded images
Live Webcam Mode → Liveness + verification combo

📊 6. Visualization & Debug Tools

Draws face mesh
Shows bounding boxes
Displays EAR values
Shows similarity score
Provides verification result

🛠️ Tech Stack

Computer Vision
OpenCV
Mediapipe (FaceMesh)
CVZone FaceDetector
Deep Learning / Recognition
DeepFace (ArcFace, RetinaFace)
TensorFlow / Keras
EmbeddingUtils (custom)

ML Algorithms

Cosine Similarity
Euclidean Distance
Liveness Detection (EAR)
Local Binary Patterns (LBP)

Frontend

Streamlit UI for interactive use
Supports live camera feed
Supports uploaded images

🔧 How to Run

1. Install Dependencies

pip install deepface mediapipe opencv-python cvzone streamlit joblib numpy tensorflow

2. Run Streamlit App

streamlit run frontend.py

📌 Use Cases

This system can be used for:

Authentication systems
Visitor verification
Attendance automation
Secure access control
Anti-spoofing verification
Identity matching for HR / offices

🧠 Why This Project Is Industry-Level

This project showcases:

✔ Live computer vision
✔ Multi-model integration
✔ Embeddings & similarity metrics
✔ Anti-spoofing techniques
✔ End-to-end system architecture
✔ Production-style class-based structure
✔ Streamlit frontend deployment

This is the exact combination recruiters and companies look for in real AI projects.

👨‍💻 Author

Joel Zerubabel
AI/ML Developer
📧 Email: jzzerubabel@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/joel-zerubabel
🐙 GitHub: https://github.com/JOEL-ZERUBABEL
