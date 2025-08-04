# 🤟 ASL Sign Language Detection



This repository contains a comprehensive pipeline for detecting and classifying American Sign Language (ASL) hand signs using deep learning and computer vision techniques. It features scripts and Jupyter notebooks for **data collection**, **model training**, and **real-time ASL sign detection** via webcam.



---



## 📖 About



**ASL Sign Language Detection** is an open-source project aimed at making sign language recognition accessible for educational, accessibility, and prototyping purposes. By collecting sign images, training deep learning models, and running real-time detection, this project demonstrates a practical workflow for ASL recognition.



Whether you’re a researcher, student, or developer, this repository provides:

- **End-to-end workflow** for ASL hand sign detection

- **Sample scripts for data collection and preprocessing**

- **Notebook for model training and evaluation**

- **Real-time detection script using your webcam**

- **Reference images and organized datasets for reproducibility**



---



## 🌟 Project Overview



**Context:**  

American Sign Language is a vital communication tool for the Deaf and hard-of-hearing communities. Automating ASL sign recognition can enhance accessibility in digital interfaces and assistive technologies.



**Objective:**  

Offer an open-source, reproducible pipeline for detecting ASL alphabet signs using image classification, with code for data collection, model training, and real-time deployment.



---



## ✨ Features



- **Data Collection:**  

  - Scripted capture of hand sign images via webcam (`Data_Collection.py`)

  - Reference images for correct pose guidance



- **Data Organization:**  

  - Scripts for splitting and organizing images into training/validation/test sets



- **Model Training:**  

  - Jupyter Notebook (`Model_Training.ipynb`) for training an image classifier (e.g., CNN) on collected sign images



- **Real-Time Detection:**  

  - Python script (`Real_Time_Detection.py`) for live ASL sign detection via webcam



- **Pre-trained Model Support:**  

  - Option to use your own trained model or provided sample model (see `Model/`)



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



- **Python 3.x**

- **Jupyter Notebook**

- **OpenCV** (for image capture and processing)

- **TensorFlow / Keras** (for deep learning model training)

- **NumPy, Pandas** (data handling)

- **Matplotlib/Seaborn** (visualization)



---



## 🚀 Installation and Setup



To run this project locally:



### 1. Clone the Repository



```bash

git clone https://github.com/ashishpant31/ASLSignLanguageDetection.git

cd ASLSignLanguageDetection

```



### 2. Create a Virtual Environment (Recommended)



```bash

python -m venv venv

```

- **Windows:** `.\venv\Scripts\activate`

- **macOS/Linux:** `source venv/bin/activate`



### 3. Install Dependencies



```bash

pip install -r Requirements.txt

```

If you encounter any missing packages, manually install them:

```bash

pip install opencv-python tensorflow keras numpy pandas matplotlib

```



### 4. Collect ASL Sign Images



Run the data collection script to capture hand sign images via webcam:



```bash

python Data_Collection.py

```

- Follow on-screen prompts to capture images for each ASL sign.

- Images are saved in `Sign_Image_48x48/`.



### 5. Organize and Split Data



Organize your collected images into training, validation, and test sets:



```bash

python Split.py

```

- This will create the directory `Split_Image_48x48/` with proper splits.



### 6. Train the Model



Open the Jupyter notebook and run through the training steps:



```bash

jupyter notebook Model_Training.ipynb

```

- The notebook will guide you through data loading, model creation, training, and evaluation.

- Trained models are saved in the `Model/` directory.



### 7. Run Real-Time ASL Detection



Start the real-time detection script using your trained model:



```bash

python Real_Time_Detection.py

```

- Ensure your webcam is connected.

- The script loads the trained model from `Model/` and performs live predictions.



---



## 💡 Usage



- **Data Collection:** Follow instructions in `Data_Collection.py` to build your own dataset.

- **Model Training:** Modify and run `Model_Training.ipynb` to experiment with different architectures or hyperparameters.

- **Real-Time Detection:** Use `Real_Time_Detection.py` for live predictions, or adapt the code for your own applications.



---

<!--

## 🖼️ Screenshots



> *Add screenshots or GIFs of the data collection process, model training results, and real-time detection here!*



---

-->

## 📋 Evaluation Criteria



- **Accuracy:** Model’s ability to correctly classify ASL signs

- **Real-time Performance:** Speed and accuracy of live detection

- **Usability:** Clear code structure and instructions

- **Extensibility:** Easy to add more signs or adapt for other hand gesture tasks



---



## 🤝 Contributing



Contributions are welcome!

- Fork this repository, create a feature branch, commit your changes, and open a pull request.



---



## 📄 License



This project is licensed under the MIT License. See the `LICENSE` file for details.



---



## 📧 Contact



For feedback, questions, or collaborations, please open an [issue](https://github.com/ashishpant31/ASLSignLanguageDetection/issues).



---



## ⚠️ .gitignore Best Practices



When using version control, **do not commit your virtual environment (`venv/`) or generated files (like datasets, model checkpoints)**. Use a `.gitignore` that includes:



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



See [GitHub’s .gitignore templates](https://github.com/github/gitignore) for more examples.



---
