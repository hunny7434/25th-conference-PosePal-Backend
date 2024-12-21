import streamlit as st
import cv2
from PIL import Image
import tempfile
from utils.process_frame_utils import process_frame_with_model

st.set_page_config(
    page_title="Feedback Chat",
    layout="wide",  # 전체 화면 너비로 설정
    initial_sidebar_state="collapsed",
)

# 상태 전환 및 페이지 이동을 위한 콜백 함수
def toggle_camera():
    st.session_state.camera_active = not st.session_state.camera_active
    # 카메라가 꺼질 때 페이지를 2로 전환
    if not st.session_state.camera_active and st.session_state.video_path:
        st.session_state.page = 2

def first_page():
    st.header("1. Select an Exercise and Use the Camera")

    # 운동 선택
    exercise = st.selectbox("Select an exercise:", ["Saree", "Push-Up", "Squat", "Lunge"])
    st.write(f"You selected: {exercise}")

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
        st.write("Camera is active. Click 'End Camera' to stop recording.")

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

                # 모델로 프레임 처리
                processed_frame = process_frame_with_model(frame)

                # 프레임을 RGB로 변환 후 Streamlit에 표시
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                st_frame.image(frame_pil, caption="Real-time Video", use_container_width=True)

                # 비디오 저장 설정
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 코덱 설정
                    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    st.session_state.video_path = temp_video_file.name
                    video_writer = cv2.VideoWriter(temp_video_file.name, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

                    if not video_writer.isOpened():
                        print("Error: VideoWriter failed to initialize.")
                    else:
                        print(f"VideoWriter initialized successfully. Saving to: {st.session_state.video_path}")

                # 원본 프레임 저장
                video_writer.write(frame)

        finally:
            print("Debug: Releasing resources...")
            cap.release()
            if video_writer:
                video_writer.release()
            print("Debug: Checking resource release...")
            if not cap.isOpened():
                print("Debug: Camera resource released successfully.")
            else:
                print("Debug: Camera resource is still open.")
            if video_writer:
                print("Debug: VideoWriter released successfully.")
            else:
                print("Debug: VideoWriter was not initialized or is already released.")