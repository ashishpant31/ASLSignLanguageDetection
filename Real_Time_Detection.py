import cv2
import numpy as np
import streamlit as st
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Define your model architecture
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(13, activation='softmax'))

# Load the weights
model.load_weights("Model\ASL_Sign_Language_Detection_Model_48x48.h5")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

label = ['A', 'B', 'I', 'L', 'M', 'N', 'S', 'T', 'U', 'V', 'W', 'Y', 'blank']

def main():
    st.title("The Silent Interpreter: ASL Detection and Translation")
    st.sidebar.markdown("## Sign Language Detection")
    
    # Create a dropdown list with options
    option = st.sidebar.selectbox("Choose an option:", ("About app", "Run on Image", "Run on Video"), index=0)

    if option == "About app":

        st.markdown("")
        st.markdown("<h5>The inability to communicate effectively poses a significant challenge for the Deaf and hard of hearing (D/HH) community. American Sign Language (ASL), a rich visual language, serves as a vital means of communication. However, real-time understanding for those unfamiliar with ASL often relies on skilled interpreters, creating accessibility limitations.<h5>", unsafe_allow_html=True)
        st.markdown("")

        st.markdown('<h2 style="color: cyan">The Need for Technological Solutions<h2>', unsafe_allow_html=True)
        st.markdown("<h5>Current solutions like sign language interpreters, while valuable, can be limited by factors like cost, scheduling, and geography. Existing technological solutions might be expensive, require specialized hardware, or lack real-time functionality, hindering widespread adoption.<h5>", unsafe_allow_html=True)
        
        st.markdown('<h2 style="color: aqua">Working of this application<h2>', unsafe_allow_html=True)
        st.markdown("<h5>1. This application offers two main functionalities: image-based ASL detection and video-based ASL detection. You can either upload an image containing an ASL sign or use your webcam to detect ASL signs in real-time.<h5>", unsafe_allow_html=True)
        st.markdown("<h5>2. Developed with Streamlit and OpenCV, The Silent Interpreter aims to bridge communication gaps and enhance accessibility for the deaf and hard-of-hearing community.<h5>", unsafe_allow_html=True)

        st.markdown('<h2 style="color: sky_blue">Social Media<h2>', unsafe_allow_html=True)
        st.link_button("Github", "https://github.com/ashishpant31", disabled=False,use_container_width=True)
        st.link_button("LinkedIn", "https://www.linkedin.com/in/ashishpant31/", disabled=False,use_container_width=True)

    def main():
        st.title("The Silent Interpreter: ASL Detection and Translation")
        option = st.sidebar.selectbox("Choose an option:", ("Run on Video",), index=0)

    if option == "Run on Video":
        video_option = st.radio("Select video source:", ("Webcam", "Pre-recorded"), index=0)

        if video_option == "Webcam":
            cap = cv2.VideoCapture(0)
        elif video_option == "Pre-recorded":
            # You can upload a video file using file_uploader
            uploaded_video = st.file_uploader("Upload a pre-recorded video:", type=["mp4", "avi", "mov"])
            if uploaded_video is not None:
                video_bytes = uploaded_video.read()
                cap = cv2.VideoCapture(io.BytesIO(video_bytes))
            else:
                st.warning("Please upload a video file.")
                return

        stframe = st.empty()

        while True:
            _, frame = cap.read()
            if frame is None:
                break

            cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
            cropframe = frame[40:300, 0:300]
            cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
            cropframe = cv2.resize(cropframe, (48, 48))
            cropframe = extract_features(cropframe)
            pred = model.predict(cropframe)
            prediction_label = label[pred.argmax()]
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
            if prediction_label == 'blank':
                cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                accu = "{:.2f}".format(np.max(pred) * 100)
                cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            stframe.image(frame, channels="BGR", width=600)


    elif option == "Run on Image":
        st.subheader("Upload an image")
        uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"], help="Upload an image of ASL sign language.", key="image_uploader")

        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, (48, 48))
            processed_image = extract_features(resized_image)
            pred = model.predict(processed_image)
            prediction_label = label[pred.argmax()]
            accu = "{:.2f}".format(np.max(pred)*100)
            st.write(f'Prediction: {prediction_label} with <span style="color:green">{accu}%</span> accuracy', unsafe_allow_html=True)
            st.image(image, caption='Uploaded Image', channels='BGR', width=300)

# Call the main function
if __name__ == "__main__":
    main()