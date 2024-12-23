import streamlit as st
from PIL import Image
from utils.process_frame_utils import process_frame_with_model
from utils.report_utils import *
import tempfile

st.set_page_config(
    layout="wide",  # 전체 화면 너비로 설정
    )

# Add this function to your script to include the CSS
def add_custom_css():
    css = """
    <style>
        /* Center the title */
        .centered-title {
            text-align: center;
            margin-top: 0;
            margin-bottom: 1rem;
            font-size: 3rem;
            font-weight: bold;
            color: #ff4b2b;
            text-shadow: 2px 2px 4px #000000;
        }
        /* General page styling */
        .main {
            background-color: #f5f5f5;
            font-family: 'Arial', sans-serif;
            text-align: left;
        }

        /* Header styling */
        h1, h2, h3 {
            color: #333333;
        }

        h1 {
            margin-bottom: 1rem;
            font-size: 2.5rem;
        }

        h2 {
            margin-top: 1rem;
            font-size: 2rem;
        }

        /* Buttons */
        .stButton>button {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 1rem;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            color: #ffffff;
        }

        /* Selectbox */
        .stSelectbox div[data-baseweb="select"] {
            background-color: #f9f9f9;
            border-radius: 5px;
        }

        /* File uploader styling */
        .stFileUploader {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 1rem;
            margin-top: 1rem;
        }

        /* Video player */
        .stVideo {
            margin: 20px auto;
            display: block;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Divider */
        hr {
            border: none;
            border-top: 1px solid #cccccc;
            margin: 2rem 0;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call this function at the beginning of your app
add_custom_css()


# 상태 전환 및 페이지 이동을 위한 콜백 함수
def toggle_camera():
    st.session_state.camera_active = not st.session_state.camera_active
    # 카메라가 꺼질 때 페이지를 2로 전환
    if not st.session_state.camera_active and st.session_state.video_path:
        st.session_state.page = 2

def click_diagnosis():
    if "video_path" in st.session_state and st.session_state.video_path:
        st.session_state.page = 2

    else:
        st.warning("Please upload or record a video first.")

def first_page():
    st.header("🏋️‍♂️ 운동 선택하고 카메라 사용하기")

    if "exercise" not in st.session_state:
        st.session_state.exercise = "Side-Lateral-Raise"

    # 운동 선택
    exercise = st.selectbox("운동 종류 선택:", ["Side-Lateral-Raise", "Push-Up", "Squat", "Lunge"])

    st.session_state.exercise = exercise
    print(f"You selected: {st.session_state.exercise}")

    # 세션 상태 초기화
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "page" not in st.session_state:
        st.session_state.page = 1

    # 버튼 생성 및 상태 전환
    if st.button(
        "Start Camera" if not st.session_state.camera_active else "End Camera",
        on_click=toggle_camera,
    ):
        # 상태 변경에 따라 Streamlit이 자동으로 UI를 갱신
        pass

    # 카메라 활성화 상태 처리
    if st.session_state.camera_active:
        print("Camera is active. Click 'End Camera' to stop recording.")

        # 카메라 설정
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open the camera.")
            return

        st_frame = st.empty()  # 빈 프레임을 Streamlit에 생성
        video_writer = None

        # 프레임 처리
        try:
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video.")
                    break

                flipped_frame = cv2.flip(frame, 1)

                # 모델로 프레임 처리
                processed_frame = process_frame_with_model(flipped_frame)

                # 프레임을 RGB로 변환 후 Streamlit에 표시
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                st_frame.image(frame_pil, caption="Real-time Video", use_container_width=True)

                # 비디오 저장 설정
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 코덱 설정
                    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    st.session_state.video_path = temp_video_file.name
                    video_writer = cv2.VideoWriter(temp_video_file.name, fourcc, 20.0, (flipped_frame.shape[1], flipped_frame.shape[0]))

                    if not video_writer.isOpened():
                        print("Error: VideoWriter failed to initialize.")
                    else:
                        print(f"VideoWriter initialized successfully. Saving to: {st.session_state.video_path}")

                # 원본 프레임 저장
                video_writer.write(flipped_frame)

        finally:
            cap.release()
            if video_writer:
                video_writer.release()
    
    # 추가된 부분: 비디오 업로드 섹션
    # 비디오 업로드 옵션을 제공
    st.write("---")
    st.header("📤 운동 영상 업로드")

    uploaded_file = st.file_uploader("Upload your video file:", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            st.session_state.video_path = temp_file.name
            
        st.video(st.session_state.video_path)
        
    # 추가된 부분: 진단하기 버튼
    st.write("---")

    if st.button("진단하기", on_click=click_diagnosis):
        # 상태 변경에 따라 Streamlit이 자동으로 UI를 갱신
        pass