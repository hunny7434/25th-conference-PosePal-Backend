from utils.test import *
from utils.model.model import *
import pickle
import os
from openai import OpenAI
from typing import List

# GPT API 설정
client = OpenAI(api_key="your api key")

model_path = "utils/model/lateralraise_fin.pkl"  # 모델 경로 설정
        
def run_posture_model(video_path):
    # Placeholder: 실제 모델을 로드하고 비디오를 분석하는 로직을 구현하세요.
    # 예시로 간단한 문자열을 반환합니다.
    smoothed_data = process_video_and_smooth(video_path)
    segments = segment_reps(smoothed_data)
    input_data = combine_segments(segments)
    
    predicted_label = inference(model_path, input_data)
    
    feedback_report = "video_path: " + video_path + "\n운동 자세가 전반적으로 양호합니다. 약간의 개선이 필요한 부분은 다음과 같습니다: 팔과 다리의 정렬을 더 신경 써주세요."
    return feedback_report, predicted_label

# Define the function to generate a report
def make_report(exercise: List[str]) -> str:
    # Construct the prompt for feedback generation
    prompt = f"""
    당신은 전문 피트니스 코치입니다. 1개의 reps마다 제공되는 운동 설명을 참고하여, 전체 세트에 대한 통합적인 피드백을 작성하세요.

    운동 목록:
    {", ".join(exercise)}

    중점을 두어야 할 부분:
    1. 운동 전반의 장점 식별.
    2. 전체 세트에서 공통으로 발견되는 문제점 또는 개선이 필요한 부분.
    3. 전체적인 자세 유지 및 부상 방지를 위한 팁.

    피드백 예제:
    "당신의 스쿼트 자세는 전반적으로 우수합니다! 척추의 중립이 잘 지켜지고 있고 무릎을 구부렸을 때 무릎의 가장 앞쪽 부분이 발가락을 넘어가지 않습니다. 다만 reps가 진행됨에 따라 등이 살짝 굽어지는 경향이 있습니다. 등이 굽어지지 않도록 시선을 정면보다 살짝 위쪽에 두면 완벽한 자세가 될 것 같습니다!"

    출력은 구체적이고 실행 가능하며 격려적인 톤을 유지해주세요.
    """

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
