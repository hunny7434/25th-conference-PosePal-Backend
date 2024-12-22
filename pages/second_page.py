import streamlit as st
import asyncio
from utils.report_utils import run_posture_model
from utils.chat_utils import chat_with_feedback, async_stream_chat_with_feedback

def second_page():
    st.header("2. Feedback Report and Chat")

    # 피드백 리포트 생성
    feedback_report = run_posture_model(st.session_state.video_path, st.session_state.exercise)

    # 세션 상태 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # 채팅 기록 저장 리스트
    if "new_message" not in st.session_state:
        st.session_state.new_message = None  # 새 메시지 상태

    # 레이아웃 분할: 왼쪽은 리포트, 오른쪽은 챗봇
    col1, col2 = st.columns(2)

    # 리포트 표시 (왼쪽)
    with col1:
        st.subheader("Feedback Report")
        st.write(feedback_report)

    # 채팅 인터페이스 (오른쪽)
    with col2:
        st.subheader("Chat about your feedback")

        # 채팅 기록 표시 영역
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            for message in st.session_state.chat_history:
                with st.chat_message("user" if message["is_user"] else "assistant"):
                    st.markdown(message["text"])

        # 마지막 메시지용 동적 컨테이너
        last_message_placeholder = st.empty()

        # 입력창
        user_input = st.chat_input("Type your message:")
        if user_input:
            # 새 메시지 저장
            st.session_state.new_message = user_input
            st.session_state.chat_history.append({"is_user": True, "text": st.session_state.new_message})

            with chat_placeholder.container():
                for message in st.session_state.chat_history:
                    with st.chat_message("user" if message["is_user"] else "assistant"):
                        st.markdown(message["text"])

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