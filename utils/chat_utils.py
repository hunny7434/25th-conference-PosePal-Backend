import torch
import openai
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI


api_key = "your api key"

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

embedding = OpenAIEmbeddings(openai_api_key=api_key)



with open("exercise_domain_knowledge.txt", "r", encoding="utf-8") as file:
    documents = file.readlines()

    #creating vecotr store
vector_store = FAISS.from_texts(documents, embedding)
vector_store.save_local("exercise_knowledge_store")

# Initialize LangChain Retrieval-based QA
qa_chain = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key), 
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)


def chat_with_feedback(report:str, chat_history, user_input:str):

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

    chat_history_str = ""
    for message in chat_history:
        if message["is_user"]:
            chat_history_str += f"User: {message['text']}\n"
        else:
            chat_history_str += f"Assistant: {message['text']}\n"


    
    # # Create a context-rich prompt combining report and user input
    # system_context = f"""You are a knowledgeable fitness assistant. Use the following exercise report as context for answering the user's questions. Keep your responses focused on the user's specific situation and previous feedback.
    # Exercise Report: {report}"""
    prompt = (
        f"당신은 운동 전문가입니다. 다음은 사용자가 한 운동에 관한 정리입니다: "
        f"'{report}'. 이 정보를 참고하여 운동에 대한 지식을 바탕으로 사용자의 질문에 답해주세요. "
        f" 다음은 이전 대화 내용입니다:'\n{chat_history_str}\n'"
        f"사용자의 질문은 다음과 같습니다: '{user_input}'. "
        f"질문에 대해 친절하고 자세하게 설명해주세요. 사용자가 이해할 수 있도록 구체적이고 명확한 답변을 해주세요."
    )

    # Use LangChain to generate a domain-specific answer
    response = qa_chain.invoke(prompt)
    answer = response['result']
    #source_documents = response['source_documents']
    

    return answer


# if __name__ == "__main__":
#     # Simulate classifier results as summaries
#     exercise_summary = ['올바른 스쿼트 자세', '상체가 너무 앞으로 기울어짐', '무릎이 발끝보다 앞으로 나감']
#     user_input = "나 무릎이 아픈데, 사이드 레터럴할때 무릎에도 무리가 가?"
#     report= "첫번째 랩에서는 올바른 사이드 레터럴 자세였어요, 두번째 랩에서는 상체가 너무 앞으로 기울어져있어요, 마지막 랩에서는 무릎이 발끝보다 앞으로 나가있어요."
#     chat_history=[]
#     # Generate chatbot response
#     response = chat_with_feedback(user_input,chat_history, report)
#     print(response)
