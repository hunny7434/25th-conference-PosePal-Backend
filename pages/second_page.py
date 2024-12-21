import streamlit as st
from utils.report_utils import run_posture_model
from utils.chat_utils import chat_with_feedback

def second_page():
    st.header("2. Feedback Report and Chat")

    # 피드백 리포트 생성
    feedback_report = run_posture_model(st.session_state.video_path)

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
        chat_placeholder = st.empty()  # 동적 컨테이너 생성
        with chat_placeholder.container():
            for message in st.session_state.chat_history:
                with st.chat_message("user" if message["is_user"] else "assistant"):
                    st.markdown(message["text"])

        # 하단 고정된 채팅 입력창
        user_input = st.chat_input("Type your message:")  # 입력창은 기본적으로 하단에 고정
        if user_input:
            # 새 메시지 상태 저장
            st.session_state.new_message = user_input

    # 새 메시지 처리
    if st.session_state.new_message:
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"is_user": True, "text": st.session_state.new_message})

        # GPT 응답 생성 및 추가
        response = chat_with_feedback(feedback_report, st.session_state.chat_history, st.session_state.new_message)
        st.session_state.chat_history.append({"is_user": False, "text": response})

        # 새 메시지 상태 초기화
        st.session_state.new_message = None

        # 메시지 UI 업데이트
        with chat_placeholder.container():
            for message in st.session_state.chat_history:
                with st.chat_message("user" if message["is_user"] else "assistant"):
                    st.markdown(message["text"])


    # "Go Back" 버튼 콜백 함수
    def go_back():
        st.session_state.clear()
        st.session_state.page = 1

    # "Go Back" 버튼 생성 및 콜백 함수 연결
    st.button("Go Back", on_click=go_back)

    # # 페이지 하단에 동영상 표시
    # st.subheader("Uploaded Video")
    # # 비디오 표시 영역
    # video_container = st.empty()
    #
    # # 동영상 파일 확인 및 업데이트
    # if st.session_state.video_path:
    #     video_container.video(st.session_state.video_path)  # 동영상을 표시
    # else:
    #     video_container.write("No video available.")  # 동영상이 없을 때 메시지 표시