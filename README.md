# Face-Track-Attendance-System

A real-time face recognition based attendance system built using Python, OpenCV, Machine Learning, and Flask.  
The system detects faces, identifies registered users, and automatically marks attendance with date and time.

---

## Features
- Real-time face detection and recognition  
- Automatic attendance recording  
- Web interface using Flask  
- Stores attendance in Excel  
- Secure user registration and dataset creation  
- Uses pre-trained deep-learning face models  
- Simple to run on any system  

---

## Tech Stack
Python, Flask, OpenCV, dlib/FaceNet, Scikit-learn, SQLite, HTML/CSS/JS

---

## Project Structure
Face-Track-Attendance-System/
│── app.py
│── requirements.txt
│── face_detection.html
│── templates/
│── static/
│── Screenshots/
└── models/   (download separately – see below)

---

## Download Model Files (Required)

Model files are stored externally because they are large.

Download from Google Drive:  
https://drive.google.com/drive/folders/1wWwiWUuL0igPD82qPw08sCntFAXzbXuG?usp=sharing

After downloading, create a folder named: models  

Place all downloaded files inside it, for example:

models/
- facenet_model.pth  
- attendance_model.pkl  
- embeddings.pkl  
- face_recognition_model-weights_manifest.json  

---

## Installation
python -m venv venv  
venv\Scripts\activate  
pip install -r requirements.txt  

---

## Run the Application
python app.py  

Open in your browser:  
http://127.0.0.1:5000  

---

## Notes
- Do not upload real faces or attendance data publicly.  
- Keep model files outside GitHub.  
- Use Google Drive or Git LFS for large files.  

---

## Contribution
Feel free to fork, enhance, and submit pull requests.
