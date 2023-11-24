from onnx_deploy import *
import streamlit as st
import time 
import tempfile

st.write("""### Face Recognition Test""")

# Upload a video file
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(video_file.read())
           
    # Open the video file for reading
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Check if the video file is open
    if cap.isOpened():
        st.write("Video Playback:")
        fps = 0
        fpss = []
        prev_time = 0
        curr_time = 0
        fps_out = st.empty()
        output_result = st.empty()
        image_out = st.empty()
        stop_button = st.button('Stop')

        # Read and display frames from the video
        while True:
            prev_time = time.time()
            ret, frame = cap.read()
            if not ret or stop_button:
                break

            frame = cv2.resize(frame, (width, height))

             # Perform face detection
            test = inference(frame)

            # Display the frame in Streamlit
            image_out.image(test, channels="BGR", use_column_width=True)
            # time.sleep(0.015)
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            fps_out.write(f"FPS:{fps}")

        # Release everything after the job is finished
        cap.release()
        # out.release()
        cv2.destroyAllWindows()
    else:
        st.write("Error: Unable to open the video file.")
else:
    st.write("Please upload a video file to display.")

