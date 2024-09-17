import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import av
import queue
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration, WebRtcMode

# Define the video processor class for real-time webcam
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, batch_size=1, conf_threshold=0.25, frame_skip=1):
        self.model = YOLO("model.pt")
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.frame_skip = frame_skip
        self.frame_count = 0  # To track frames for frame skipping
        self.result_queue = queue.Queue()  # Initialize the queue here

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform object detection with the specified confidence threshold and batch size
        results = self.model(img, conf=self.conf_threshold, batch=self.batch_size)

        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Queue the results for later processing
        self.result_queue.put(results[0])

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

def main():
    st.title("YOLO Object Detection with Streamlit")

    # Sidebar for user selection
    st.sidebar.title("Options")
    option = st.sidebar.selectbox(
        "Select input type:", ("Real-time Camera", "Upload Image", "Upload Video")
    )

    batch_size = st.sidebar.slider(
        "Batch Size", min_value=1, max_value=32, value=1, step=1
    )
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05
    )
    frame_skip = st.sidebar.number_input(
        "Frame Skip (Set to 1 for no skipping)", min_value=1, max_value=10, value=1
    )

    if option == "Real-time Camera":
        st.write("Real-time object detection using webcam")
        
        # Use WebRTC for real-time video streaming
        webrtc_ctx = webrtc_streamer(
            key="yolo",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_processor_factory=lambda: YOLOVideoProcessor(
                batch_size=batch_size,
                conf_threshold=conf_threshold,
                frame_skip=frame_skip,
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Debug statement to check the state of webrtc_ctx
        st.write(f"WebRTC context state: {webrtc_ctx.state}")

        if st.checkbox("Show the detected labels", value=True):
            if webrtc_ctx.state.playing:
                labels_placeholder = st.empty()
                while True:
                    # Debug statement to check if video_processor is None
                    st.write(f"Video processor: {webrtc_ctx.video_processor}")
                    
                    if webrtc_ctx.video_processor is not None:
                        if not webrtc_ctx.video_processor.result_queue.empty():
                            result = webrtc_ctx.video_processor.result_queue.get()
                            # Assuming result is a dataframe or similar
                            labels_placeholder.table(result.pandas().to_dict())
                    else:
                        st.write("Video processor is None")

    elif option == "Upload Image":
        st.write("Upload an image for object detection")
        uploaded_image = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            results = YOLO("model.pt")(image, conf=conf_threshold, batch=batch_size)
            annotated_image = results[0].plot()
            st.image(annotated_image, channels="BGR", caption="Processed Image")

    elif option == "Upload Video":
        st.write("Upload a video for object detection")
        uploaded_video = st.file_uploader(
            "Choose a video...", type=["mp4", "mov", "avi"]
        )

        if uploaded_video is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_video.read())
                temp_video_path = tfile.name

            cap = cv2.VideoCapture(temp_video_path)
            st_frame = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = YOLO("model.pt")(frame, conf=conf_threshold, batch=batch_size)
                annotated_frame = results[0].plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(annotated_frame_rgb, channels="RGB")

            cap.release()

if __name__ == "__main__":
    main()
