from utils.test import *
from utils.model.model import *
from openai import OpenAI
from typing import List

from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
api_key = os.getenv("GPT_API_KEY")


# GPT API 설정
client = OpenAI(api_key=api_key)
exercise_model_map = {
    "Side-Lateral-Raise": "utils/model/lateralraise_fin.pkl",
    "Lunge": "utils/model/lunge_fin.pkl"
}
        
def run_posture_model(video_path, exercise):

    # 영상 분석
    smoothed_data = process_video_and_smooth(video_path, exercise)
    segments = segment_reps(smoothed_data)
    input_data = combine_segments(segments)
    
    
    ## 레포트에 보일 사진을 위한 파트
    first_dataframe = input_data[0]
    peak_frame_number = first_dataframe.iloc[0]['frame_no']
    user_image = extract_frame_as_image(video_path, peak_frame_number)
    output_image = process_pose_comparison(f"utils/gt_images/{exercise}.jpg", user_image)
    
    # 인퍼런스
    model_path = exercise_model_map[exercise]  # 모델 경로 설정
    predicted_label = inference(model_path, input_data, exercise)

    # exercise = {'377': '올바른 사이드 레터럴 레이즈 자세', '378': '무릎 반동이 있는 사이드 레터럴 레이즈 자세','379': '어깨를 으쓱하는 사이드 레터럴 레이즈 자세'
    #          ,'380': '상완과 전완의 각도 고정이 안 된 사이드 레터럴 레이즈 자세', '381': '손목의 각도가 고정이 되지 않은 사이드 레터럴 레이즈 자세'
    #          , '382': '상체 반동이 있는 사이드 레터럴 레이즈 자세'}

    # result = [exercise[i] for i in predicted_label]
    report = make_report(predicted_label, exercise)
    
    # report = "video_path: " + video_path + "\n운동 자세가 전반적으로 양호합니다. 약간의 개선이 필요한 부분은 다음과 같습니다: 팔과 다리의 정렬을 더 신경 써주세요."

    # # 현재 스크립트의 디렉토리 경로
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(current_dir, "test.png")  # 파일 경로 생성

    # # 이미지 읽기
    # image = cv2.imread(file_path)

    # if image is None:
    #     raise FileNotFoundError(f"Error: Could not read the image file at {file_path}")

    return report, output_image

# Define the function to generate a report
def make_report(predictions: List[List], exercise) -> str:
    
    
    
    flattened_exercise = {code for set_codes in predictions for code in set_codes}
    
    # Construct the prompt for feedback generation
    
    prompt = exercise_prompts[exercise].format(predictions=flattened_exercise)
    
    # Send the prompt to GPT using the latest API format
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 유능한 피트니스 코치입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )

    # Return the generated response
    return response.choices[0].message.content


exercise_prompts = {
    "Side-Lateral-Raise": """
    당신은 전문 피트니스 코치입니다. 숫자 코드 형태로 1개의 reps마다 제공되는 사용자 운동 피드백을 참고하여, 전체 세트에 대한 통합적인 피드백을 작성하세요.

    다음은 숫자 코드와 사이드 레터럴 레이즈 자세 상태의 매핑 정보입니다:
    - 377: 모든 조건(무릎 반동 없음, 어깨 으쓱 없음, 상완과 전완 각도 고정, 손목 각도 고정, 상체 반동 없음)을 만족.
    - 378: 무릎 반동 있음, 나머지 조건은 만족.
    - 379: 어깨 으쓱 있음, 나머지 조건은 만족.
    - 380: 상완과 전완 각도가 고정되지 않음, 나머지 조건은 만족.
    - 381: 손목 각도가 고정되지 않음, 나머지 조건은 만족.
    - 382: 상체 반동 있음, 나머지 조건은 만족.
    
    숫자 코드:
    {predictions}

    중점을 두어야 할 부분:
    1. 운동 전반의 장점 식별.
    2. 전체 세트에서 공통으로 발견되는 문제점 또는 개선이 필요한 부분.
    3. 전체적인 자세 유지 및 부상 방지를 위한 팁.

    아래 형식에 맞춰서 출력해줘: 
    당신의 사이드 레터럴 레이즈 점수는?

    **종합 점수**: XX점 / 100점

    ### 항목별 점수
    1. **무릎 반동**: XX점 / 20점  
    - **피드백**: 

    2. **어깨 으쓱**: XX점 / 20점  
    - **피드백**:

    3. **상완과 전완 각도 고정**: XX점 / 20점  
    - **피드백**: 

    4. **손목 각도 고정**: XX점 / 20점  
    - **피드백**:

    5. **상체 안정성**: XX점 / 20점  
    - **피드백**: 

    ---

    ### **총평**
    
    
    **추천 개선 방안**:
    """,
    
    "Lunge": """
    당신은 전문 피트니스 코치입니다. 숫자 코드 형태로 1개의 reps마다 제공되는 사용자 운동 피드백을 참고하여, 전체 세트에 대한 통합적인 피드백을 작성하세요.

    다음은 숫자 코드와 런지 상태의 매핑 정보입니다:
    - 81: 모든 조건(앞다리 무릎 각도 90도, 몸통 방향 및 무릎 정렬, 뒷다리 무릎 각도 90도, 척추의 중립, 상체의 과도한 숙임/젖힘 없음)을 만족.
    - 82: 앞다리 무릎 각도 90도 아님, 나머지 조건은 만족.
    - 83: 몸통 방향 및 무릎 정렬 안 됨, 나머지 조건은 만족.
    - 84: 뒷다리 무릎 각도 90도 아님, 나머지 조건은 만족.
    - 85: 척추의 중립 없음, 나머지 조건은 만족.
    - 86: 상체의 과도한 숙임/젖힘 있음, 나머지 조건은 만족.
    
    숫자 코드:
    {predictions}

    중점을 두어야 할 부분:
    1. 운동 전반의 장점 식별.
    2. 전체 세트에서 공통으로 발견되는 문제점 또는 개선이 필요한 부분.
    3. 전체적인 자세 유지 및 부상 방지를 위한 팁.

    아래 형식에 맞춰서 출력해줘: 
    당신의 런지 점수는?

    **종합 점수**: XX점 / 100점

    ### 항목별 점수
    1. **앞다리 무릎 각도 90도**: XX점 / 20점  
    - **피드백**: 

    2. **몸통 방향 및 무릎 정렬**: XX점 / 20점  
    - **피드백**:

    3. **뒷다리 무릎 각도 90도**: XX점 / 20점  
    - **피드백**: 

    4. **척추의 중립**: XX점 / 20점  
    - **피드백**:

    5. **상체의 과도한 숙임/젖힘 여부**: XX점 / 20점
    - **피드백**: 

    ---

    ### **총평**
    
    
    **추천 개선 방안**:
    """
}
