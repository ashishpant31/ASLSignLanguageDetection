import cv2
import numpy as np
import streamlit as st
import io
import tempfile
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="The Silent Interpreter: ASL Detection",
    page_icon="🤟",
    layout="wide",
)

# ==================== Custom CSS for a dark theme and better look ====================
st.markdown("""
    <style>
    /* Main App Background and Text Colors for Dark Theme */
    .stApp {
        background-color: #1a1a1a;
        color: #f0f2f6;
    }

    /* Header Styles */
    .main-header {
        font-size: 50px !important;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
        text-shadow: 2px 2px 4px #000;
    }
    .subheader-main {
        font-size: 28px;
        color: #007BFF;
        text-align: center;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        color: #f0f2f6;
    }
    
    /* Info Box and General Text */
    .stAlert {
        background-color: #262730;
        color: #f0f2f6;
    }
    .stAlert p {
        color: #f0f2f6;
    }
    /* st.container styling for the boxes */
    .stContainer {
        border: 1px solid #444;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #262730;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    
    /* Other elements to ensure they are dark-theme compatible */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f2f6;
    }
    a {
        color: #007BFF;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Model and Utility Functions ====================

@st.cache_resource
def load_model():
    """
    Loads the Keras model and its weights, caching it to prevent
    re-instantiation on every Streamlit rerun.
    """
    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        Conv2D(512, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        Conv2D(512, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(13, activation='softmax')
    ])
    
    try:
        model.load_weights("Model/ASL_Sign_Language_Detection_Model_48x48.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.info("Please ensure 'ASL_Sign_Language_Detection_Model_48x48.h5' is in the 'Model' directory.")
        return None

# Load the model using the cached function
model = load_model()

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

label = ['A', 'B', 'I', 'L', 'M', 'N', 'S', 'T', 'U', 'V', 'W', 'Y', 'blank']

# ==================== Main App Logic ====================
def run_on_video():
    st.header("Real-Time ASL Detection")
    video_option = st.radio("Select video source:", ("Webcam", "Pre-recorded"), index=0)

    cap = None
    if video_option == "Webcam":
        cap = cv2.VideoCapture(0)
    elif video_option == "Pre-recorded":
        uploaded_video = st.file_uploader("Upload a pre-recorded video:", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_video.read())
                video_path = temp_file.name
            cap = cv2.VideoCapture(video_path)
        else:
            st.warning("Please upload a video file.")
            return

    stframe = st.empty()
    stop_button = st.button("Stop Detection")

    if cap and cap.isOpened():
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Processing frame
            cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
            cropframe = frame[40:300, 0:300]
            cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
            cropframe = cv2.resize(cropframe, (48, 48))
            
            processed_frame = extract_features(cropframe)
            pred = model.predict(processed_frame, verbose=0)
            prediction_label = label[np.argmax(pred)]
            
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
            
            if prediction_label == 'blank':
                cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                accu = "{:.2f}".format(np.max(pred) * 100)
                cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            stframe.image(frame, channels="BGR", width=600)
    
        cap.release()
        st.success("Detection stopped.")
    elif video_option == "Webcam":
        st.error("Webcam not found or cannot be opened. Please check your camera settings and permissions.")


def run_on_image():
    st.header("ASL Detection from an Image")
    uploaded_image = st.file_uploader(
        "Upload an image of an ASL sign:",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # FIX: Set a smaller, fixed width to prevent over-zooming and keep content visible
        st.image(image, caption='Uploaded Image', channels='BGR', width=250)
        
        st.subheader("Prediction")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (48, 48))
        processed_image = extract_features(resized_image)
        
        pred = model.predict(processed_image, verbose=0)
        prediction_label = label[np.argmax(pred)]
        accu = "{:.2f}".format(np.max(pred) * 100)
        
        st.info(f'Prediction: **{prediction_label}** with **{accu}%** accuracy')

def about_app():
    # Header
    st.markdown("<h1 class='main-header'>The Silent Interpreter</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='subheader-main'>ASL Detection and Translation</h2>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

    # Box 1: Introduction to Sign Language and the Problem
    st.container()
    st.subheader("1. The Importance of Sign Language")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        American Sign Language (ASL) is a complete, natural language with its own grammar and syntax. It is the primary means of communication for millions of Deaf and hard-of-hearing (D/HH) individuals in North America. Despite its richness, a significant communication barrier persists between the D/HH community and the hearing world.
        
        This barrier limits access to information, educational opportunities, and social interactions. While professional interpreters are invaluable, they are not always available, affordable, or scalable for everyday situations.
        """)
    with col2:
        st.image("https://github.com/ashishpant31/ASL-Sign-Language-Detection-using-CNN/raw/main/ASL%20Sign%20Language%20Detection.gif", 
                 caption="Real-time ASL sign detection in action", width=300)
    
    st.markdown("<hr/>", unsafe_allow_html=True)

    # Box 2: Our Project's Mission and Solution
    st.container()
    st.subheader("2. Our Mission: Bridging the Communication Gap")
    st.markdown("""
    This project, "The Silent Interpreter," aims to leverage the power of modern technology—specifically, computer vision and deep learning—to create an accessible and real-time solution. Our goal is to develop a tool that can instantly recognize and translate ASL signs into text, thereby empowering D/HH individuals and facilitating seamless communication with those who are not fluent in sign language.
    
    By providing an intuitive, real-time application, we hope to create a more inclusive environment for everyone.
    """)
    st.info("The application supports both live webcam feeds and pre-recorded videos/images, making it versatile for various use cases.")

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Box 3: Technical Overview: How It Works
    st.container()
    st.subheader("3. Technical Deep Dive: The CNN Model")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        The core of this application is a custom-built Convolutional Neural Network (CNN) model. The process for sign recognition involves several key steps:
        
        1.  **Image Preprocessing**: Raw input (webcam feed or image) is converted to grayscale and resized to a consistent `48x48` pixel resolution.
        2.  **Feature Extraction**: The CNN, with its multiple layers of convolutions and pooling, automatically learns to identify and extract the most relevant features of the hand gestures from the image.
        3.  **Classification**: The extracted features are passed to a dense neural network that classifies the sign into one of the 13 supported ASL signs (`A`, `B`, `I`, `L`, `M`, `N`, `S`, `T`, `U`, `V`, `W`, `Y`) or a `blank` state.
        """)
    with col4:
        st.markdown("""
        **The model architecture includes:**
        -   Four `Conv2D` layers with increasing filter sizes (`128, 256, 512, 512`).
        -   `MaxPooling2D` layers after each convolution to reduce dimensionality.
        -   `Dropout` layers to prevent overfitting.
        -   A `Flatten` layer to prepare data for the fully connected layers.
        -   Multiple `Dense` layers to perform the final classification, concluding with a `softmax` activation function for multi-class prediction.
        
        This architecture is optimized for accuracy and efficiency in real-time scenarios.
        """)
    
    st.markdown("<hr/>", unsafe_allow_html=True)

    # Box 4: Future Potential
    st.container()
    st.subheader("4. Future Enhancements")
    st.markdown("""
    While this application is a significant step, the future holds potential for even greater impact. Possible enhancements include:
    -   Expanding the model to recognize the full ASL alphabet, numbers, and common phrases.
    -   Integrating text-to-speech functionality to provide an auditory output alongside the text.
    -   Improving model robustness to handle varying lighting conditions and backgrounds.
    -   Developing a mobile-friendly version for on-the-go use.
    
    This project is a foundation for more advanced tools that can truly empower the D/HH community.
    """)

def main():
    # Sidebar for navigation
    st.sidebar.markdown("<h3 class='sidebar-title'>Sign Language Detection</h3>", unsafe_allow_html=True)
    option = st.sidebar.selectbox(
        "Choose an option:",
        ("About App", "Run on Image", "Run on Video"),
        index=0,
    )

    if option == "About App":
        about_app()
    elif option == "Run on Image":
        if model:
            run_on_image()
    elif option == "Run on Video":
        if model:
            run_on_video()

if __name__ == "__main__":
    main()