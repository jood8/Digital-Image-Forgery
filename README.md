# Digital-Image-Forgery
Digital Image Forgery Detection
📌 Project Overview
This project is an advanced forensic tool developed as a graduation requirement for Computer Engineering (2026). It distinguishes between Original Digital Images and Recaptured Images (photos of screens) using a hybrid feature extraction approach (Texture & Frequency analysis) and an SVM classifier.

The system integrates Explainable AI (XAI) using SHAP to provide transparency, showing exactly which features influenced the model's decision.

🚀 Key Features
Dual Feature Extraction: Uses Local Binary Patterns (LBP) for texture and Fast Fourier Transform (FFT) for frequency domain analysis.

Interactive Dashboard: A user-friendly interface built with Streamlit.

XAI Transparency: Real-time SHAP visualizations to explain each prediction.

Forensic Tools: Integrated LBP and FFT spectrum visualizers.

Performance: Achieved an accuracy of ~81.82% with a detailed comparison of multiple ML algorithms.

📂 Project Structure
Plaintext
├── app.py              # Main Streamlit Dashboard
├── model.pkl           # Trained SVM Classifier
├── requirements.txt    # Project Dependencies
├── README.md           # Project Documentation
└── data_samples/       # Representative samples of the dataset (Original/Recaptured)
📊 Dataset Access
Due to GitHub's file size and count limits, the full structured dataset used for training and testing is hosted on Google Drive.

Drive Link: https://drive.google.com/file/d/1B2EruL7bXb-GBPwkSKQIMXdHb9FHXaND/view?usp=sharing
Full Dataset Link : https://drive.google.com/drive/folders/1ZgRJnekUx8hCwCjFSds_qefi23Vdv_mv?usp=sharing
Sample Data: A small subset of images is available in the /data_samples folder in this repository for quick reference.

🛠️ Tech Stack
Language: Python 

Framework: Streamlit

Core Libraries: scikit-learn, shap, opencv-python, matplotlib, numpy, pandas.

🎯 How to Run Locally
Clone the repository:  

Install dependencies: pip install -r requirements.txt

Run the app , streamlit run app(https://digital-image-forgery-mb2boa3amwyfhbqgzzxvgl.streamlit.app/)
📺 Demo Video
A full video demonstration of the system's features and performance can be viewed here: https://drive.google.com/file/d/1eK8KEYUIFkIU9Cd5MjMn8irc3MipWKKR/view?usp=sharing

👤 Author
Eng Jood Shatnawi
Irbid, Jordan# Digital-Image-Forgery
