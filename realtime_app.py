import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import av
import queue
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration

# Define the video processor class for real-time webcam
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, batch_size=1, conf_threshold=0.25, frame_skip=1):
        self.model = YOLO("model.pt")
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.result_queue = queue.Queue()  # Initialize the queue for storing detection results

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform object detection with the specified confidence threshold and batch size
        results = self.model(img, conf=self.conf_threshold, batch=self.batch_size)

        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Put results in the queue for later display
        detections = results[0].pandas().xyxy[0].to_dict(orient="records")
        self.result_queue.put(detections)

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


def process_video(uploaded_video, conf_threshold, batch_size, model):
    # Temporary file to store the video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_video.read())
        temp_video_path = tfile.name

    # Open the video using OpenCV
    cap = cv2.VideoCapture(temp_video_path)

    st_frame = st.empty()

    # Read and process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection with the specified confidence threshold and batch size
        results = model(frame, conf=conf_threshold, batch=batch_size)

        # Annotate the frame with bounding boxes
        annotated_frame = results[0].plot()

        # Convert the frame back to RGB for display in Streamlit
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the frame in the Streamlit app
        st_frame.image(annotated_frame_rgb, channels="RGB")

    cap.release()


def main():
    st.title("YOLO Object Detection with Streamlit")

    # Sidebar for user selection
    st.sidebar.title("Options")
    option = st.sidebar.selectbox(
        "Select input type:", ("Real-time Camera", "Upload Image", "Upload Video")
    )

    # Batch size and confidence threshold input
    batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=32, value=1, step=1)
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

    # Frame skipping option (optional)
    frame_skip = st.sidebar.number_input("Frame Skip (Set to 1 for no skipping)", min_value=1, max_value=10, value=1)

    # Initialize YOLO model
    model = YOLO("model.pt")

    if option == "Real-time Camera":
        st.write("Real-time object detection using webcam")

        # WebRTC configuration with public TURN server
        rtc_configuration = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},  # Google STUN server
                {
                    "urls": "turn:turn.anyfirewall.com:443?transport=tcp",  # Public TURN server
                    "username": "webrtc",
                    "credential": "webrtc"
                }
            ]
        })

        # WebRTC streamer for real-time object detection
        webrtc_ctx = webrtc_streamer(
            key="yolo",
            video_processor_factory=lambda: YOLOVideoProcessor(
                batch_size=batch_size,
                conf_threshold=conf_threshold,
                frame_skip=frame_skip,
            ),
            rtc_configuration=rtc_configuration,
        )

        # Display detected labels if checkbox is selected
        if st.checkbox("Show the detected labels", value=True):
            if webrtc_ctx.video_processor:
                result_queue = webrtc_ctx.video_processor.result_queue
                labels_placeholder = st.empty()
                while True:
                    detections = result_queue.get()
                    labels_placeholder.table(detections)

    elif option == "Upload Image":
        st.write("Upload an image for object detection")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            results = model(image, conf=conf_threshold, batch=batch_size)
            annotated_image = results[0].plot()
            st.image(annotated_image, channels="BGR", caption="Processed Image")

    elif option == "Upload Video":
        st.write("Upload a video for object detection")
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

        if uploaded_video is not None:
            st.write("Processing video...")
            process_video(uploaded_video, conf_threshold, batch_size, model)


if __name__ == "__main__":
    main()
