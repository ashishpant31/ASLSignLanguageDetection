# 🤟 ASL Sign Language Detection

This repository provides a complete pipeline for detecting and classifying American Sign Language (ASL) hand signs using deep learning and computer vision. It includes scripts for **data collection**, **model training**, and **real-time ASL sign detection** via webcam.

---

## 📖 About

**ASL Sign Language Detection** is designed to make sign language recognition accessible for learning, accessibility, and prototyping. It covers:

- **End-to-end workflow** for ASL hand sign detection
- **Sample scripts for data collection and preprocessing**
- **Notebook for model training and evaluation**
- **Real-time detection script using webcam**
- **Reference images and organized datasets**

---

## 🌟 Project Overview

**Objective:**  
Provide an open-source, reproducible workflow for recognizing ASL alphabet signs using image classification.

---

## ✨ Features

- **Data Collection:** Capture hand sign images via webcam.
- **Data Organization:** Split and organize images into training, validation, and test sets.
- **Model Training:** Train an image classifier (CNN) on sign images.
- **Real-Time Detection:** Run live ASL sign detection via webcam.
- **Pre-trained Model Support:** Use your own or a sample model.

---

## 📂 Project Structure

```
ASLSignLanguageDetection/
├── Data_Collection.py
├── Hand_Signs_Reference.png
├── Model_Training.ipynb
├── Real_Time_Detection.py
├── Requirements.txt
├── Split.py
├── Model/                  # Saved model(s)
├── Sign_Image_48x48/       # Raw collected images
├── Split_Image_48x48/      # Organized train/val/test splits
├── Testing/                # Test images
└── README.md
```

---

## 🛠️ Technologies Used

- Python 3.x
- Jupyter Notebook
- OpenCV
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- MediaPipe
- scikit-learn
- pillow

---

## 🚀 Installation and Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ashishpant31/ASLSignLanguageDetection.git
   cd ASLSignLanguageDetection
   ```

2. **(Optional but Recommended) Create a Virtual Environment**
   ```bash
   python -m venv venv
   # Windows: .\venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r Requirements.txt
   ```

4. **Collect ASL Sign Images**
   ```bash
   python Data_Collection.py
   ```
   - Follow instructions to capture images for each ASL sign.

5. **Organize and Split Data**
   ```bash
   python Split.py
   ```

6. **Train the Model**
   ```bash
   jupyter notebook Model_Training.ipynb
   ```
   - Run through the notebook to build and evaluate your model.

7. **Run Real-Time ASL Detection**
   ```bash
   python Real_Time_Detection.py
   ```

---

## 🖼️ Example Screenshots

Below are some example screenshots demonstrating various stages of the project:

### Data Collection Interface

<img width="1919" height="948" alt="Data Collection" src="https://github.com/user-attachments/assets/b686cacf-ac6f-420e-a47e-5fc06fee8e09" />

---

### Real-Time Detection Demo

<img width="1895" height="796" alt="Real Time Detection" src="https://github.com/user-attachments/assets/2e4d9fcb-684c-4139-8fb3-e68c4349f307" />
<img width="1864" height="765" alt="Real Time Detection 2" src="https://github.com/user-attachments/assets/86b21faa-978d-4d2e-99cb-cb6f5724f343" />
<img width="1884" height="746" alt="Real Time Detection 3" src="https://github.com/user-attachments/assets/04db777c-b6b7-4674-bdae-2db5b9ae5713" />

---


## 📋 Evaluation

- **Accuracy:** Model’s ability to classify ASL signs
- **Real-time Performance:** Speed and accuracy of live detection
- **Usability:** Clear instructions and code structure
- **Extensibility:** Easy to add more signs or adapt for other gestures

---

## 🤝 Contributing

Contributions are welcome!  
Fork this repository, create a feature branch, commit your changes, and open a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## ⚠️ .gitignore Best Practices

Add the following to `.gitignore`:

```
venv/
*.pyc
__pycache__/
.ipynb_checkpoints/
.env
.DS_Store
*.h5
*.csv
```

---
