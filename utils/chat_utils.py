from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict

def chat_with_feedback(report: str, chat_history: List[Dict[str, str]], user_input: str) -> str:

    # Placeholder: 실제 GPT 모델과 연동하여 응답을 생성하는 로직을 구현하세요.
    # 예시로 간단한 문자열을 반환합니다.

    """
    Processes user input with context from exercise feedback report and chat history to generate appropriate responses.

    Args:
        report (str): Exercise feedback report containing analysis of user's form
        chat_history (List[Dict[str, str]]): List of previous chat messages between user and bot
        user_input (str): Current user message/question

    Returns:
        tuple: (response, updated_chat_history)
    """
    # Initialize the LLM
    llm = ChatOllama(model="llama2", base_url="http://localhost:11434")

    # Create a context-rich prompt combining report and user input
    system_context = f"""You are a knowledgeable fitness assistant. Use the following exercise report as context for answering the user's questions. Keep your responses focused on the user's specific situation and previous feedback.
    Exercise Report: {report}"""

    # Add the system context as the first message if chat history is empty
    if not chat_history:
        chat_history.append({"is_user": False, "text": system_context})

    # Add the new user input to chat history
    chat_history.append({"is_user": True, "text": user_input})

    # Prepare the full conversation context
    messages = []
    for message in chat_history:
        if message["is_user"]:
            messages.append(HumanMessage(content=message["text"]))
        else:
            messages.append(AIMessage(content=message["text"]))

    # Generate response using the LLM
    response_content = llm.invoke(messages, stream=False)

    # Extract only the content from the response
    response = response_content.content

    # Add the assistant's response to chat history
    chat_history.append({"is_user": False, "text": response})

    return response