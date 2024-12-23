import cv2
import streamlit as st
import asyncio
from PIL import Image
from utils.report_utils import run_posture_model
from utils.chat_utils import async_stream_chat_with_feedback


def second_page():
    # st.header("운동 결과 분석 📊")
    st.markdown(
        """
        <h2 style="text-align: center; font-size: 40px; margin-top: 20px;">
            운동 결과 분석 📊
        </h2>
        """,
        unsafe_allow_html=True,
    )
    st.write("---")
    # 기존 화면을 모두 지우고 로딩 메시지 표시
    loading_placeholder = st.empty()  # 로딩 화면용 자리 표시자

    with loading_placeholder.container():
        st.markdown(
            """
            <div style="text-align: center; font-size: 20px;">
                Analyzing your exercise... Please wait ⏳
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 피드백 리포트 생성
    feedback_report, feedback_image = run_posture_model(st.session_state.video_path, st.session_state.exercise)

    # 로딩 메시지 제거
    loading_placeholder.empty()

    # 세션 상태 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # 채팅 기록 저장 리스트
    if "new_message" not in st.session_state:
        st.session_state.new_message = None  # 새 메시지 상태

    # 레이아웃 분할: 왼쪽은 리포트, 오른쪽은 챗봇
    col1, col2 = st.columns(2)

    # 리포트 표시 (왼쪽)
    with col1:
        st.subheader("분석 리포트📄")
        st.write(feedback_report)

        st.subheader("상세 프레임 이미지")
        try:
            # OpenCV 이미지를 RGB로 변환 (Streamlit 호환)
            image_rgb = cv2.cvtColor(feedback_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)  # NumPy 배열을 PIL 이미지로 변환

            # Streamlit에 이미지 표시
            st.image(pil_image, caption="Test Image", use_container_width=True)
        except FileNotFoundError as e:
            print(str(e))

    # 채팅 인터페이스 (오른쪽)
    with col2:
        st.subheader("💬 챗봇과 대화하기")

        # 채팅 기록 표시 영역
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            for message in st.session_state.chat_history:
                with st.chat_message("user" if message["is_user"] else "assistant"):
                    st.markdown(message["text"])

        # 마지막 메시지용 동적 컨테이너
        last_message_placeholder = st.empty()

        # 입력창
        user_input = st.chat_input("운동 자세에 대해 챗봇과 대화하기:")
        if user_input:
            # 새 메시지 저장
            st.session_state.new_message = user_input
            st.session_state.chat_history.append({"is_user": True, "text": st.session_state.new_message})

            with chat_placeholder.container():
                for message in st.session_state.chat_history:
                    with st.chat_message("user" if message["is_user"] else "assistant"):
                        st.markdown(message["text"])

            st.session_state.chat_history.append({"is_user": False, "text": ""})
            with last_message_placeholder.container():
                with st.chat_message("assistant"):
                    st.markdown("")

            async def stream_response():
                response_stream = async_stream_chat_with_feedback(
                    feedback_report,
                    st.session_state.chat_history,
                    st.session_state.new_message
                )

                assistant_response = ""
                async for chunk in response_stream:
                    assistant_response += chunk
                    st.session_state.chat_history[-1]["text"] = assistant_response

                    # UI 업데이트: 마지막 메시지만 갱신
                    with last_message_placeholder.container():
                        with st.chat_message("assistant"):
                            st.markdown(assistant_response)

                # 메시지 상태 초기화
                st.session_state.new_message = None

            # 비동기 처리 실행
            asyncio.run(stream_response())

    # "Go Back" 버튼 콜백 함수
    def go_back():
        st.session_state.clear()
        st.session_state.page = 1

    # "Go Back" 버튼 생성 및 콜백 함수 연결
    st.button("Go Back", on_click=go_back)